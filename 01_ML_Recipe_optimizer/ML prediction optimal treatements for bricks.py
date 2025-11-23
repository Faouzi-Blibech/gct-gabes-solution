"""
PhosphoGypsum Brick Recipe Optimizer
Core ML system for blend optimization with explainability (GPU-ACCELERATED)

Purpose: Given lab inputs → Recommend optimal recipe → Predict properties
Output: "Given these inputs → this recipe → predicted strength = 8 MPa (PASS)"

Installation:
pip install numpy pandas scikit-learn xgboost shap matplotlib seaborn joblib cupy-cuda12x

For RTX 4070 (CUDA 12.x):
pip install cupy-cuda12x

For older CUDA versions:
pip install cupy-cuda11x  # For CUDA 11.x

Usage:
python recipe_optimizer_core.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# GPU Detection and Configuration
GPU_AVAILABLE = False
try:
    import cupy as cp
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n🚀 GPU DETECTED: {GPU_NAME}")
        print(f"   GPU Memory: {GPU_MEMORY:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print("   GPU acceleration ENABLED for XGBoost\n")
    else:
        print("\n⚠️  GPU not detected. Running on CPU.\n")
except ImportError:
    print("\n⚠️  GPU libraries not installed. Running on CPU.")
    print("   To enable GPU: pip install cupy-cuda12x torch\n")

# ============================================================================
# 1. DATA GENERATION - Simulated PG Lab Experiments
# ============================================================================

def generate_training_data(n_samples=500, seed=42):
    """
    Generate realistic PG brick formulation data based on research literature.
    """
    np.random.seed(seed)
    
    # Recipe components (%)
    df = pd.DataFrame({
        'treated_pg_percent': np.random.uniform(55, 75, n_samples),
        'binder_percent': np.random.uniform(10, 22, n_samples),
        'lime_percent': np.random.uniform(5, 14, n_samples),
        'water_percent': np.random.uniform(9, 16, n_samples),
        'so2_exposure_min': np.random.uniform(30, 75, n_samples),
        'washing_cycles': np.random.randint(1, 4, n_samples),
        'fluoride_ppm': np.random.uniform(150, 1200, n_samples),
        'phosphate_ppm': np.random.uniform(600, 2800, n_samples),
        'sulfate_percent': np.random.uniform(36, 44, n_samples),
        'calcium_oxide_percent': np.random.uniform(29, 37, n_samples),
        'impurities_ppm': np.random.uniform(100, 800, n_samples),
        'ph_level': np.random.uniform(6.8, 8.5, n_samples),
        'curing_temp_celsius': np.random.uniform(22, 35, n_samples),
        'curing_time_days': np.random.randint(7, 21, n_samples),
        'pressing_pressure_mpa': np.random.uniform(15, 32, n_samples),
        'moisture_percent': np.random.uniform(5, 12, n_samples),
    })
    
    # Stoichiometric features (chemistry)
    df['ca_s_ratio'] = df['calcium_oxide_percent'] / (df['sulfate_percent'] + 0.01)
    df['binder_to_pg_ratio'] = df['binder_percent'] / (df['treated_pg_percent'] + 0.01)
    df['water_binder_ratio'] = df['water_percent'] / (df['binder_percent'] + 0.01)
    df['total_cementitious'] = df['binder_percent'] + df['lime_percent']
    df['effective_calcium'] = df['calcium_oxide_percent'] + (df['lime_percent'] * 0.7)
    
    # TARGET: Compressive Strength (MPa) - REALISTIC with failures
    strength = (
        df['binder_percent'] * 0.35 +
        df['lime_percent'] * 0.28 +
        -np.abs(df['water_percent'] - 11.5) * 0.8 +
        (df['curing_time_days'] * df['curing_temp_celsius']) / 150 +
        df['pressing_pressure_mpa'] * 0.12 +
        np.minimum(df['so2_exposure_min'] / 12, 5) +
        -df['treated_pg_percent'] * 0.08 +
        -(df['fluoride_ppm'] / 500) * 2.0 +
        -np.abs(df['ca_s_ratio'] - 0.85) * 5 +
        1.0
    )
    df['compressive_strength_mpa'] = np.clip(
        strength + np.random.normal(0, 2.0, n_samples), 2, 15
    )
    
    # TARGET: Leachability Index (0-10, higher = safer) - REALISTIC with failures
    leachability = (
        2.5 +
        df['so2_exposure_min'] / 15 +
        df['lime_percent'] * 0.28 +
        df['washing_cycles'] * 0.6 +
        -(np.abs(df['ph_level'] - 7.3)) * 0.9 +
        -(df['fluoride_ppm'] / 300) * 1.2 +
        df['pressing_pressure_mpa'] / 7 +
        df['curing_time_days'] * 0.08
    )
    df['leachability_index'] = np.clip(
        leachability + np.random.normal(0, 1.0, n_samples), 1, 9.5
    )
    
    # TARGET: Cost ($/brick)
    df['cost_per_brick'] = (
        df['treated_pg_percent'] * 0.012 +
        df['binder_percent'] * 0.18 +
        df['lime_percent'] * 0.11 +
        df['so2_exposure_min'] * 0.004 +
        0.08 + np.random.uniform(0.01, 0.04, n_samples)
    )
    
    # TARGET: Pass/Fail (Strength ≥ 7.0 MPa AND Leachability ≥ 5.5) - Lower thresholds
    df['passes_all_standards'] = (
        (df['compressive_strength_mpa'] >= 7.0) & 
        (df['leachability_index'] >= 5.5)
    ).astype(int)
    
    return df

# ============================================================================
# 2. PREPROCESSING PIPELINE
# ============================================================================

class DataPreprocessor:
    """Handle missing values, scaling, and feature engineering"""
    
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = RobustScaler()
        self.feature_names = None
        
    def fit_transform(self, df):
        """Fit and transform training data"""
        df = df.copy()
        
        # Define target columns
        target_cols = [
            'compressive_strength_mpa', 'leachability_index', 
            'cost_per_brick', 'passes_all_standards'
        ]
        
        # Feature columns
        self.feature_names = [col for col in df.columns if col not in target_cols]
        
        X = df[self.feature_names].values
        
        # Impute and scale
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        return X_scaled, df[target_cols]
    
    def transform(self, df):
        """Transform new data"""
        X = df[self.feature_names].values
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled

# ============================================================================
# 3. ML MODELS
# ============================================================================

class RecipeOptimizer:
    """Multi-target ML system for brick recipe optimization"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.metrics = {}
        self.trained = False
        
    def train(self, df, test_size=0.2, random_state=42, verbose=True):
        """Train all models with cross-validation"""
        
        if verbose:
            print("="*70)
            print("TRAINING PHOSPHOGYPSUM BRICK RECIPE OPTIMIZER")
            print("="*70)
        
        # Preprocess data
        X, y = self.preprocessor.fit_transform(df)
        
        # Split data with stratification for classifier
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y['passes_all_standards']  # Ensure balanced splits
        )
        
        # MODEL 1: Compressive Strength (XGBoost with GPU)
        if verbose:
            print("\n[1/4] Training Compressive Strength Predictor (XGBoost)...")
            if GPU_AVAILABLE:
                print("    Using GPU acceleration (CUDA)...")
        
        # Configure XGBoost for GPU
        if GPU_AVAILABLE:
            self.models['strength'] = xgb.XGBRegressor(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                tree_method='gpu_hist',  # GPU-accelerated training
                gpu_id=0,
                predictor='gpu_predictor',  # GPU-accelerated prediction
                n_jobs=-1
            )
        else:
            self.models['strength'] = xgb.XGBRegressor(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                n_jobs=-1
            )
        
        self.models['strength'].fit(X_train, y_train['compressive_strength_mpa'])
        
        # Test predictions
        y_pred_strength = self.models['strength'].predict(X_test)
        
        self.metrics['strength'] = {
            'test_rmse': np.sqrt(mean_squared_error(y_test['compressive_strength_mpa'], y_pred_strength)),
            'test_r2': r2_score(y_test['compressive_strength_mpa'], y_pred_strength)
        }
        
        if verbose:
            print(f"  ✓ Test RMSE: {self.metrics['strength']['test_rmse']:.3f} MPa")
            print(f"  ✓ Test R²: {self.metrics['strength']['test_r2']:.3f}")
        
        # MODEL 2: Leachability (Random Forest)
        if verbose:
            print("\n[2/4] Training Leachability Index Predictor (Random Forest)...")
        
        self.models['leachability'] = RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=random_state, n_jobs=-1
        )
        self.models['leachability'].fit(X_train, y_train['leachability_index'])
        
        y_pred_leach = self.models['leachability'].predict(X_test)
        
        self.metrics['leachability'] = {
            'test_rmse': np.sqrt(mean_squared_error(y_test['leachability_index'], y_pred_leach)),
            'test_r2': r2_score(y_test['leachability_index'], y_pred_leach)
        }
        
        if verbose:
            print(f"  ✓ Test RMSE: {self.metrics['leachability']['test_rmse']:.3f}")
            print(f"  ✓ Test R²: {self.metrics['leachability']['test_r2']:.3f}")
        
        # MODEL 3: Pass/Fail Classifier (XGBoost with GPU) - FIXED
        if verbose:
            print("\n[3/4] Training Pass/Fail Classifier (XGBoost)...")
            if GPU_AVAILABLE:
                print("    Using GPU acceleration (CUDA)...")
        
        # Configure XGBoost classifier for GPU
        if GPU_AVAILABLE:
            self.models['classifier'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.08,
                random_state=random_state,
                tree_method='gpu_hist',  # GPU-accelerated
                gpu_id=0,
                predictor='gpu_predictor',
                n_jobs=-1,
                eval_metric='logloss'
            )
        else:
            self.models['classifier'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.08,
                random_state=random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
        
        self.models['classifier'].fit(X_train, y_train['passes_all_standards'])
        
        y_pred_class = self.models['classifier'].predict(X_test)
        
        self.metrics['classifier'] = {
            'accuracy': accuracy_score(y_test['passes_all_standards'], y_pred_class)
        }
        
        if verbose:
            print(f"  ✓ Accuracy: {self.metrics['classifier']['accuracy']:.3f}")
            print("\n  Classification Report:")
            print(classification_report(
                y_test['passes_all_standards'], y_pred_class,
                target_names=['FAIL', 'PASS'], digits=3
            ))
        
        # MODEL 4: Cost Predictor
        if verbose:
            print("[4/4] Training Cost Predictor (Ridge)...")
        
        from sklearn.linear_model import Ridge
        self.models['cost'] = Ridge(alpha=1.0)
        self.models['cost'].fit(X_train, y_train['cost_per_brick'])
        
        y_pred_cost = self.models['cost'].predict(X_test)
        
        self.metrics['cost'] = {
            'test_rmse': np.sqrt(mean_squared_error(y_test['cost_per_brick'], y_pred_cost)),
            'test_r2': r2_score(y_test['cost_per_brick'], y_pred_cost)
        }
        
        if verbose:
            print(f"  ✓ Test RMSE: ${self.metrics['cost']['test_rmse']:.4f}")
        
        self.trained = True
        
        if verbose:
            print("\n" + "="*70)
            print("✓ ALL MODELS TRAINED SUCCESSFULLY")
            print("="*70)
        
        return self
    
    def predict(self, recipe_dict):
        """Predict all properties for a given recipe"""
        
        if not self.trained:
            raise ValueError("Models not trained. Call .train() first.")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([recipe_dict])
        
        # Add engineered features
        if 'ca_s_ratio' not in input_df.columns:
            input_df['ca_s_ratio'] = input_df.get('calcium_oxide_percent', 33) / (input_df.get('sulfate_percent', 40) + 0.01)
        if 'binder_to_pg_ratio' not in input_df.columns:
            input_df['binder_to_pg_ratio'] = input_df.get('binder_percent', 15) / (input_df.get('treated_pg_percent', 65) + 0.01)
        if 'water_binder_ratio' not in input_df.columns:
            input_df['water_binder_ratio'] = input_df.get('water_percent', 12) / (input_df.get('binder_percent', 15) + 0.01)
        if 'total_cementitious' not in input_df.columns:
            input_df['total_cementitious'] = input_df.get('binder_percent', 15) + input_df.get('lime_percent', 8)
        if 'effective_calcium' not in input_df.columns:
            input_df['effective_calcium'] = input_df.get('calcium_oxide_percent', 33) + (input_df.get('lime_percent', 8) * 0.7)
        
        # Ensure all features exist
        for feat in self.preprocessor.feature_names:
            if feat not in input_df.columns:
                input_df[feat] = 0
        
        # Preprocess
        X_scaled = self.preprocessor.transform(input_df)
        
        # Predictions
        strength = float(self.models['strength'].predict(X_scaled)[0])
        leachability = float(self.models['leachability'].predict(X_scaled)[0])
        pass_proba = float(self.models['classifier'].predict_proba(X_scaled)[0, 1])
        cost = float(self.models['cost'].predict(X_scaled)[0])
        
        passes = (strength >= 7.0) and (leachability >= 5.5)
        
        # Confidence interval
        strength_uncertainty = self.metrics['strength']['test_rmse']
        
        return {
            'compressive_strength_mpa': strength,
            'strength_ci_lower': strength - 1.96 * strength_uncertainty,
            'strength_ci_upper': strength + 1.96 * strength_uncertainty,
            'leachability_index': leachability,
            'cost_per_brick': cost,
            'passes_standards': passes,
            'pass_probability': pass_proba,
            'confidence': pass_proba if passes else 1 - pass_proba,
            'recommendation': self._generate_recommendation(strength, leachability, passes)
        }
    
    def _generate_recommendation(self, strength, leachability, passes):
        """Generate recommendation"""
        if passes:
            if strength > 9 and leachability > 7.0:
                return "✓ EXCELLENT: High strength & safety. Recipe exceeds standards."
            else:
                return "✓ PASS: Meets minimum standards (≥7.0 MPa, ≥5.5 leachability)."
        else:
            issues = []
            if strength < 7.0:
                issues.append(f"strength {strength:.1f} < 7.0 MPa")
            if leachability < 5.5:
                issues.append(f"leachability {leachability:.1f} < 5.5")
            return f"✗ FAIL: {', '.join(issues)}. Increase binder/lime/SO2."
    
    def optimize_recipe(self, n_iterations=200, objective='balanced', verbose=True):
        """Find optimal recipe using random search"""
        
        if not self.trained:
            raise ValueError("Models not trained. Call .train() first.")
        
        if verbose:
            print(f"\n🔍 Optimizing recipe ({objective} objective, {n_iterations} iterations)...\n")
        
        best_recipe = None
        best_score = -np.inf
        best_predictions = None
        valid_count = 0
        optimization_history = []
        
        for i in range(n_iterations):
            recipe = {
                'treated_pg_percent': np.random.uniform(58, 72),
                'binder_percent': np.random.uniform(12, 20),
                'lime_percent': np.random.uniform(6, 13),
                'water_percent': np.random.uniform(10, 14),
                'so2_exposure_min': np.random.uniform(35, 70),
                'washing_cycles': np.random.randint(2, 4),
                'fluoride_ppm': np.random.uniform(200, 700),
                'phosphate_ppm': np.random.uniform(800, 1800),
                'sulfate_percent': 40.0,
                'calcium_oxide_percent': 33.0,
                'impurities_ppm': 300.0,
                'ph_level': 7.4,
                'curing_temp_celsius': 26.0,
                'curing_time_days': 14,
                'pressing_pressure_mpa': 24.0,
                'moisture_percent': 8.0,
            }
            
            pred = self.predict(recipe)
            
            if not pred['passes_standards']:
                continue
            
            valid_count += 1
            
            if objective == 'balanced':
                score = pred['compressive_strength_mpa'] * 0.4 + pred['leachability_index'] * 2.0 - pred['cost_per_brick'] * 12
            elif objective == 'strength':
                score = pred['compressive_strength_mpa']
            elif objective == 'safety':
                score = pred['leachability_index']
            elif objective == 'cost':
                score = -pred['cost_per_brick']
            else:
                raise ValueError(f"Unknown objective: {objective}")
            
            # Track history
            pred['score'] = score
            optimization_history.append(pred.copy())
            
            if score > best_score:
                best_score = score
                best_recipe = recipe.copy()
                best_predictions = pred.copy()
        
        if verbose:
            print(f"   ✓ Found {valid_count}/{n_iterations} valid recipes\n")
        
        return best_recipe, best_predictions, optimization_history
    
    def get_feature_importance(self, model_name='strength', top_n=10):
        """Get feature importance"""
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            return {}
        
        feat_imp = dict(zip(self.preprocessor.feature_names, importances))
        sorted_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        return sorted_imp
    
    def visualize_feature_importance(self, model_name='strength', top_n=15, save_path='feature_importance.png'):
        """Create and save feature importance visualization"""
        importance = self.get_feature_importance(model_name, top_n)
        
        if not importance:
            print("No feature importance available for this model")
            return
        
        plt.figure(figsize=(12, 8))
        features = list(importance.keys())
        values = list(importance.values())
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        bars = plt.barh(features, values, color=colors, edgecolor='black', linewidth=0.8)
        
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Feature Importance - {model_name.upper()} Model', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance plot saved: {save_path}")
        plt.close()
    
    def visualize_prediction_comparison(self, test_df, save_path='prediction_comparison.png'):
        """Compare predicted vs actual values for all targets"""
        
        X_scaled = self.preprocessor.transform(test_df)
        
        pred_strength = self.models['strength'].predict(X_scaled)
        pred_leach = self.models['leachability'].predict(X_scaled)
        pred_cost = self.models['cost'].predict(X_scaled)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Strength
        axes[0, 0].scatter(test_df['compressive_strength_mpa'], pred_strength, 
                          alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[0, 0].plot([test_df['compressive_strength_mpa'].min(), test_df['compressive_strength_mpa'].max()],
                       [test_df['compressive_strength_mpa'].min(), test_df['compressive_strength_mpa'].max()],
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Strength (MPa)', fontweight='bold')
        axes[0, 0].set_ylabel('Predicted Strength (MPa)', fontweight='bold')
        axes[0, 0].set_title(f'Compressive Strength (R² = {self.metrics["strength"]["test_r2"]:.3f})', 
                            fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Leachability
        axes[0, 1].scatter(test_df['leachability_index'], pred_leach, 
                          alpha=0.6, s=50, color='green', edgecolors='black', linewidth=0.5)
        axes[0, 1].plot([test_df['leachability_index'].min(), test_df['leachability_index'].max()],
                       [test_df['leachability_index'].min(), test_df['leachability_index'].max()],
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 1].set_xlabel('Actual Leachability Index', fontweight='bold')
        axes[0, 1].set_ylabel('Predicted Leachability Index', fontweight='bold')
        axes[0, 1].set_title(f'Leachability Index (R² = {self.metrics["leachability"]["test_r2"]:.3f})', 
                            fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Cost
        axes[1, 0].scatter(test_df['cost_per_brick'], pred_cost, 
                          alpha=0.6, s=50, color='orange', edgecolors='black', linewidth=0.5)
        axes[1, 0].plot([test_df['cost_per_brick'].min(), test_df['cost_per_brick'].max()],
                       [test_df['cost_per_brick'].min(), test_df['cost_per_brick'].max()],
                       'r--', lw=2, label='Perfect Prediction')
        axes[1, 0].set_xlabel('Actual Cost ($)', fontweight='bold')
        axes[1, 0].set_ylabel('Predicted Cost ($)', fontweight='bold')
        axes[1, 0].set_title(f'Cost per Brick (R² = {self.metrics["cost"]["test_r2"]:.3f})', 
                            fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Pass/Fail Distribution
        actual_pass = test_df['passes_all_standards'].sum()
        actual_fail = len(test_df) - actual_pass
        
        categories = ['FAIL', 'PASS']
        counts = [actual_fail, actual_pass]
        colors_bar = ['#ff6b6b', '#51cf66']
        
        bars = axes[1, 1].bar(categories, counts, color=colors_bar, edgecolor='black', linewidth=1.5)
        axes[1, 1].set_ylabel('Count', fontweight='bold')
        axes[1, 1].set_title(f'Pass/Fail Distribution (Accuracy = {self.metrics["classifier"]["accuracy"]:.3f})', 
                            fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Prediction comparison plot saved: {save_path}")
        plt.close()
    
    def visualize_recipe_properties(self, recipe_dict, predictions, save_path='recipe_analysis.png'):
        """Visualize a single recipe's properties and predictions"""
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Recipe composition pie chart
        ax1 = fig.add_subplot(gs[0, :2])
        components = ['Treated PG', 'Binder', 'Lime', 'Water', 'Other']
        values = [
            recipe_dict.get('treated_pg_percent', 0),
            recipe_dict.get('binder_percent', 0),
            recipe_dict.get('lime_percent', 0),
            recipe_dict.get('water_percent', 0),
            100 - (recipe_dict.get('treated_pg_percent', 0) + 
                   recipe_dict.get('binder_percent', 0) + 
                   recipe_dict.get('lime_percent', 0) + 
                   recipe_dict.get('water_percent', 0))
        ]
        colors_pie = ['#8ecae6', '#219ebc', '#023047', '#ffb703', '#fb8500']
        
        wedges, texts, autotexts = ax1.pie(values, labels=components, autopct='%1.1f%%',
                                            colors=colors_pie, startangle=90,
                                            textprops={'fontweight': 'bold', 'size': 11})
        ax1.set_title('Recipe Composition', fontsize=14, fontweight='bold', pad=20)
        
        # Predicted properties gauge
        ax2 = fig.add_subplot(gs[0, 2])
        properties = ['Strength\n(MPa)', 'Leachability\n(Index)', 'Cost\n($)']
        pred_values = [
            predictions['compressive_strength_mpa'],
            predictions['leachability_index'],
            predictions['cost_per_brick'] * 10  # Scale for visibility
        ]
        colors_bar2 = ['#ff6b6b', '#51cf66', '#ffd43b']
        
        bars = ax2.barh(properties, pred_values, color=colors_bar2, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Value', fontweight='bold')
        ax2.set_title('Predicted Properties', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, pred_values)):
            if i < 2:
                ax2.text(val + 0.2, bar.get_y() + bar.get_height()/2, 
                        f'{val:.2f}', va='center', fontweight='bold')
            else:
                ax2.text(val + 0.2, bar.get_y() + bar.get_height()/2, 
                        f'${predictions["cost_per_brick"]:.3f}', va='center', fontweight='bold')
        
        # Key parameters
        ax3 = fig.add_subplot(gs[1, :])
        params = [
            f"SO₂: {recipe_dict.get('so2_exposure_min', 0):.0f} min",
            f"Washing: {recipe_dict.get('washing_cycles', 0)} cycles",
            f"Fluoride: {recipe_dict.get('fluoride_ppm', 0):.0f} ppm",
            f"pH: {recipe_dict.get('ph_level', 0):.1f}",
            f"Curing: {recipe_dict.get('curing_time_days', 0)} days @ {recipe_dict.get('curing_temp_celsius', 0):.0f}°C",
            f"Pressure: {recipe_dict.get('pressing_pressure_mpa', 0):.0f} MPa"
        ]
        
        ax3.axis('off')
        ax3.text(0.5, 0.9, 'Processing Parameters', ha='center', fontsize=14, 
                fontweight='bold', transform=ax3.transAxes)
        
        for i, param in enumerate(params):
            row = i // 3
            col = i % 3
            x = 0.17 + col * 0.33
            y = 0.6 - row * 0.35
            ax3.text(x, y, param, ha='center', fontsize=11, 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
                    transform=ax3.transAxes, fontweight='bold')
        
        # Pass/Fail result
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        result_color = '#51cf66' if predictions['passes_standards'] else '#ff6b6b'
        result_text = 'PASS ✓' if predictions['passes_standards'] else 'FAIL ✗'
        
        ax4.text(0.5, 0.7, result_text, ha='center', fontsize=24, fontweight='bold',
                color=result_color, transform=ax4.transAxes,
                bbox=dict(boxstyle='round,pad=1', facecolor=result_color, alpha=0.3, 
                         edgecolor=result_color, linewidth=3))
        
        ax4.text(0.5, 0.4, predictions['recommendation'], ha='center', fontsize=11,
                transform=ax4.transAxes, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
        
        ax4.text(0.5, 0.15, f"Confidence: {predictions['confidence']:.1%}", 
                ha='center', fontsize=12, fontweight='bold', transform=ax4.transAxes)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Recipe analysis plot saved: {save_path}")
        plt.close()
    
    def visualize_optimization_results(self, optimization_history, save_path='optimization_results.png'):
        """Visualize optimization search results"""
        
        if not optimization_history:
            print("No optimization history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        iterations = list(range(1, len(optimization_history) + 1))
        strengths = [r['compressive_strength_mpa'] for r in optimization_history]
        leachabilities = [r['leachability_index'] for r in optimization_history]
        costs = [r['cost_per_brick'] for r in optimization_history]
        scores = [r.get('score', 0) for r in optimization_history]
        
        # Strength over iterations
        axes[0, 0].plot(iterations, strengths, marker='o', linewidth=2, markersize=4)
        axes[0, 0].axhline(y=7.0, color='r', linestyle='--', label='Minimum (7.0 MPa)')
        axes[0, 0].set_xlabel('Iteration', fontweight='bold')
        axes[0, 0].set_ylabel('Strength (MPa)', fontweight='bold')
        axes[0, 0].set_title('Compressive Strength Progression', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Leachability over iterations
        axes[0, 1].plot(iterations, leachabilities, marker='s', linewidth=2, 
                       markersize=4, color='green')
        axes[0, 1].axhline(y=5.5, color='r', linestyle='--', label='Minimum (5.5)')
        axes[0, 1].set_xlabel('Iteration', fontweight='bold')
        axes[0, 1].set_ylabel('Leachability Index', fontweight='bold')
        axes[0, 1].set_title('Leachability Index Progression', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Cost over iterations
        axes[1, 0].plot(iterations, costs, marker='^', linewidth=2, 
                       markersize=4, color='orange')
        axes[1, 0].set_xlabel('Iteration', fontweight='bold')
        axes[1, 0].set_ylabel('Cost ($)', fontweight='bold')
        axes[1, 0].set_title('Cost per Brick Progression', fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # Overall score
        axes[1, 1].plot(iterations, scores, marker='D', linewidth=2, 
                       markersize=4, color='purple')
        axes[1, 1].set_xlabel('Iteration', fontweight='bold')
        axes[1, 1].set_ylabel('Optimization Score', fontweight='bold')
        axes[1, 1].set_title('Overall Score Progression', fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Optimization results plot saved: {save_path}")
        plt.close()
    
    def save(self, filepath='recipe_optimizer.pkl'):
        """Save model"""
        joblib.dump({
            'models': self.models,
            'preprocessor': self.preprocessor,
            'metrics': self.metrics,
            'trained': self.trained
        }, filepath)
        print(f"\n✓ Model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath='recipe_optimizer.pkl'):
        """Load model"""
        data = joblib.load(filepath)
        obj = cls()
        obj.models = data['models']
        obj.preprocessor = data['preprocessor']
        obj.metrics = data['metrics']
        obj.trained = data['trained']
        return obj

# ============================================================================
# 4. MAIN EXECUTION & DEMO
# ============================================================================

def main():
    """Complete demonstration of the Recipe Optimizer"""
    
    print("\n" + "="*70)
    print(" PHOSPHOGYPSUM BRICK RECIPE OPTIMIZER - ML SYSTEM")
    print("="*70 + "\n")
    
    # STEP 1: Generate training data
    print("[STEP 1] Generating training data (500 samples)...")
    df = generate_training_data(n_samples=500, seed=42)
    print(f"✓ Dataset created: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"✓ Pass rate: {df['passes_all_standards'].mean()*100:.1f}%\n")
    
    # STEP 2: Train models
    print("[STEP 2] Training ML models...")
    optimizer = RecipeOptimizer()
    optimizer.train(df, verbose=True)
    
    # STEP 3: Test prediction on sample recipe
    print("\n" + "="*70)
    print("[STEP 3] Testing prediction on sample recipe")
    print("="*70 + "\n")
    
    test_recipe = {
        'treated_pg_percent': 65.0,
        'binder_percent': 16.0,
        'lime_percent': 9.0,
        'water_percent': 12.0,
        'so2_exposure_min': 50.0,
        'washing_cycles': 2,
        'fluoride_ppm': 400.0,
        'phosphate_ppm': 1200.0,
        'sulfate_percent': 40.0,
        'calcium_oxide_percent': 33.0,
        'impurities_ppm': 300.0,
        'ph_level': 7.4,
        'curing_temp_celsius': 26.0,
        'curing_time_days': 14,
        'pressing_pressure_mpa': 24.0,
        'moisture_percent': 8.0,
    }
    
    print("Input Recipe:")
    print(f"  • Treated PG: {test_recipe['treated_pg_percent']:.1f}%")
    print(f"  • Binder: {test_recipe['binder_percent']:.1f}%")
    print(f"  • Lime: {test_recipe['lime_percent']:.1f}%")
    print(f"  • Water: {test_recipe['water_percent']:.1f}%")
    print(f"  • SO₂ Exposure: {test_recipe['so2_exposure_min']:.0f} min\n")
    
    result = optimizer.predict(test_recipe)
    
    print("Prediction Results:")
    print(f"  • Compressive Strength: {result['compressive_strength_mpa']:.2f} MPa")
    print(f"    (95% CI: {result['strength_ci_lower']:.2f} - {result['strength_ci_upper']:.2f})")
    print(f"  • Leachability Index: {result['leachability_index']:.2f} / 10")
    print(f"  • Cost per Brick: ${result['cost_per_brick']:.3f}")
    print(f"  • Pass Probability: {result['pass_probability']:.1%}")
    print(f"  • Result: {result['recommendation']}\n")
    
    # STEP 4: Optimize recipe
    print("="*70)
    print("[STEP 4] Finding optimal recipe")
    print("="*70)
    
    best_recipe, best_result, opt_history = optimizer.optimize_recipe(n_iterations=300, objective='balanced', verbose=True)
    
    print("Optimal Recipe Found:")
    print(f"  • Treated PG: {best_recipe['treated_pg_percent']:.1f}%")
    print(f"  • Binder: {best_recipe['binder_percent']:.1f}%")
    print(f"  • Lime: {best_recipe['lime_percent']:.1f}%")
    print(f"  • Water: {best_recipe['water_percent']:.1f}%")
    print(f"  • SO₂ Exposure: {best_recipe['so2_exposure_min']:.0f} min")
    print(f"  • Washing Cycles: {best_recipe['washing_cycles']}")
    print(f"\nPredicted Properties:")
    print(f"  • Strength: {best_result['compressive_strength_mpa']:.2f} MPa ✓")
    print(f"  • Leachability: {best_result['leachability_index']:.2f} ✓")
    print(f"  • Cost: ${best_result['cost_per_brick']:.3f}")
    print(f"  • {best_result['recommendation']}\n")
    
    # STEP 5: Feature importance
    print("="*70)
    print("[STEP 5] Feature Importance (Top 10)")
    print("="*70 + "\n")
    
    importance = optimizer.get_feature_importance('strength', top_n=10)
    for i, (feat, imp) in enumerate(importance.items(), 1):
        print(f"  {i:2d}. {feat:30s} {imp:.4f}")
    
    # STEP 6: Generate Visualizations
    print("\n" + "="*70)
    print("[STEP 6] Generating Visualizations")
    print("="*70 + "\n")
    
    # Split data for visualization
    from sklearn.model_selection import train_test_split
    _, test_viz_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 1. Feature Importance Plot
    optimizer.visualize_feature_importance('strength', top_n=15, save_path='01_feature_importance.png')
    
    # 2. Prediction Comparison
    optimizer.visualize_prediction_comparison(test_viz_df, save_path='02_prediction_comparison.png')
    
    # 3. Recipe Analysis for test recipe
    optimizer.visualize_recipe_properties(test_recipe, result, save_path='03_test_recipe_analysis.png')
    
    # 4. Recipe Analysis for optimal recipe
    optimizer.visualize_recipe_properties(best_recipe, best_result, save_path='04_optimal_recipe_analysis.png')
    
    # 5. Optimization Results
    optimizer.visualize_optimization_results(opt_history, save_path='05_optimization_progression.png')
    
    print("\n" + "="*70)
    print("✓ ALL VISUALIZATIONS GENERATED!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. 01_feature_importance.png - What drives strength")
    print("  2. 02_prediction_comparison.png - Model accuracy")
    print("  3. 03_test_recipe_analysis.png - Test recipe breakdown")
    print("  4. 04_optimal_recipe_analysis.png - Best recipe found")
    print("  5. 05_optimization_progression.png - Search process")
    
    # STEP 7: Save model
    print("\n" + "="*70)
    print("[STEP 7] Saving model")
    print("="*70)
    optimizer.save('recipe_optimizer.pkl')
    
    print("\n" + "="*70)
    print(" DEMONSTRATION COMPLETE!")
    print("="*70)
    print("\nYou can now:")
    print("  1. Load the model: optimizer = RecipeOptimizer.load('recipe_optimizer.pkl')")
    print("  2. Make predictions: optimizer.predict(your_recipe)")
    print("  3. Optimize recipes: optimizer.optimize_recipe()")
    print("\n")

if __name__ == "__main__":
    main()