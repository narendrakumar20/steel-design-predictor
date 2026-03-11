"""
Inverse Design Engine: Dream Steel Builder
Optimize composition to meet target properties using genetic algorithm
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random
from src.utils import ELEMENT_COLS, validate_composition, calculate_cost_estimate, format_composition_string

class InverseDesignEngine:
    """Generate optimal steel compositions from desired properties"""
    
    def __init__(self, predictor, feature_engineer, scaler, selected_features=None):
        """
        Args:
            predictor: Trained SteelPropertyPredictor
            feature_engineer: SteelFeatureEngineering instance
            scaler: Feature scaler from training
            selected_features: List of selected feature names from training
        """
        self.predictor = predictor
        self.feature_engineer = feature_engineer
        self.scaler = scaler
        self.selected_features = selected_features
        
        # Typical element ranges for steels (weight %)
        self.element_bounds = {
            'C': (0.01, 1.5),
            'Mn': (0.01, 2.0),
            'Si': (0.01, 1.5),
            'Cr': (0.0, 20.0),
            'Ni': (0.0, 15.0),
            'Mo': (0.0, 3.0),
            'Cu': (0.0, 1.5),
            'V': (0.0, 0.5),
            'W': (0.0, 2.0),
            'Co': (0.0, 5.0),
            'Al': (0.0, 1.0),
            'Ti': (0.0, 0.3),
            'Nb': (0.0, 0.3),
        }
    
    def _create_random_composition(self) -> Dict[str, float]:
        """Generate random valid steel composition"""
        composition = {}
        for element, (min_val, max_val) in self.element_bounds.items():
            composition[element] = random.uniform(min_val, max_val)
        
        # Ensure total doesn't exceed realistic values
        total = sum(composition.values())
        if total > 30:  # Reduce if too many alloying elements
            scale = 25 / total
            composition = {k: v * scale for k, v in composition.items()}
        
        return composition
    
    def _mutate_composition(self, composition: Dict[str, float], mutation_rate: float = 0.3) -> Dict[str, float]:
        """Apply mutation to composition"""
        new_composition = composition.copy()
        
        for element in ELEMENT_COLS:
            if random.random() < mutation_rate:
                min_val, max_val = self.element_bounds[element]
                # Gaussian mutation around current value
                delta = random.gauss(0, (max_val - min_val) * 0.1)
                new_composition[element] = np.clip(
                    new_composition[element] + delta,
                    min_val, max_val
                )
        
        return new_composition
    
    def _crossover_compositions(self, parent1: Dict[str, float], 
                               parent2: Dict[str, float]) -> Dict[str, float]:
        """Combine two compositions"""
        child = {}
        for element in ELEMENT_COLS:
            # Random weighted average
            alpha = random.random()
            child[element] = alpha * parent1[element] + (1 - alpha) * parent2[element]
        
        return child
    
    def _predict_properties(self, composition: Dict[str, float]) -> np.ndarray:
        """Predict properties for a composition"""
        # Create dataframe
        comp_df = pd.DataFrame([composition])
        
        # Engineer features
        comp_features = self.feature_engineer.create_features(comp_df)
        
        # Select the same features that were used during training
        if self.selected_features is not None:
            comp_selected = comp_features[self.selected_features]
        else:
            comp_selected = comp_features
        
        # Scale
        comp_scaled = pd.DataFrame(
            self.scaler.transform(comp_selected),
            columns=comp_selected.columns
        )
        
        # Predict
        properties = self.predictor.predict(comp_scaled)[0]
        return properties
    
    def _fitness_function(self, composition: Dict[str, float],
                         target_ys: float, target_uts: float, target_elong: float,
                         weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                         cost_penalty: float = 0.0) -> float:
        """
        Calculate fitness (lower is better)
        
        Args:
            composition: Steel composition
            target_ys/uts/elong: Target properties
            weights: Property importance weights
            cost_penalty: Penalty for expensive compositions
        """
        try:
            # Predict properties
            pred_ys, pred_uts, pred_elong = self._predict_properties(composition)
            
            # Calculate errors
            ys_error = abs(pred_ys - target_ys) / target_ys
            uts_error = abs(pred_uts - target_uts) / target_uts
            elong_error = abs(pred_elong - target_elong) / target_elong
            
            # Weighted fitness
            fitness = (
                weights[0] * ys_error +
                weights[1] * uts_error +
                weights[2] * elong_error
            )
            
            # Add cost penalty if specified
            if cost_penalty > 0:
                cost = calculate_cost_estimate(composition)
                fitness += cost_penalty * (cost / 10.0)  # Normalize cost
            
            return fitness
            
        except Exception as e:
            return float('inf')  # Invalid composition
    
    def optimize(self, 
                target_ys: float, 
                target_uts: float, 
                target_elong: float,
                population_size: int = 100,
                generations: int = 50,
                top_k: int = 5,
                weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                cost_penalty: float = 0.0,
                verbose: bool = True) -> List[Dict]:
        """
        Genetic algorithm optimization for inverse design
        
        Returns:
            List of top_k compositions with predicted properties and fitness
        """
        
        if verbose:
            print(f"\n{'='*60}")
            print("INVERSE DESIGN ENGINE: DREAM STEEL BUILDER")
            print(f"{'='*60}")
            print(f"Target Properties:")
            print(f"  Yield Strength:  {target_ys:.0f} MPa")
            print(f"  UTS:             {target_uts:.0f} MPa")
            print(f"  Elongation:      {target_elong:.1f} %")
            print(f"\nOptimization Parameters:")
            print(f"  Population:      {population_size}")
            print(f"  Generations:     {generations}")
            print(f"  Cost penalty:    {cost_penalty}")
        
        # Initialize population
        population = [self._create_random_composition() for _ in range(population_size)]
        
        best_fitness_history = []
        
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = [
                self._fitness_function(comp, target_ys, target_uts, target_elong, 
                                      weights, cost_penalty)
                for comp in population
            ]
            
            # Sort by fitness
            sorted_indices = np.argsort(fitness_scores)
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]
            
            best_fitness_history.append(fitness_scores[0])
            
            if verbose and (gen % 10 == 0 or gen == generations - 1):
                print(f"Generation {gen:3d} | Best Fitness: {fitness_scores[0]:.4f} | "
                      f"Avg Fitness: {np.mean(fitness_scores[:10]):.4f}")
            
            # Selection: Keep top 30%
            elite_size = population_size // 3
            elite = population[:elite_size]
            
            # Generate new population
            new_population = elite.copy()
            
            while len(new_population) < population_size:
                # Crossover
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                child = self._crossover_compositions(parent1, parent2)
                
                # Mutation
                child = self._mutate_composition(child, mutation_rate=0.3)
                
                new_population.append(child)
            
            population = new_population
        
        # Final evaluation of top candidates
        results = []
        for i in range(min(top_k, len(population))):
            composition = population[i]
            fitness = fitness_scores[i]
            
            # Get predictions
            pred_ys, pred_uts, pred_elong = self._predict_properties(composition)
            
            # Calculate errors
            ys_error = abs(pred_ys - target_ys) / target_ys * 100
            uts_error = abs(pred_uts - target_uts) / target_uts * 100
            elong_error = abs(pred_elong - target_elong) / target_elong * 100
            
            # Cost estimate
            cost = calculate_cost_estimate(composition)
            
            results.append({
                'rank': i + 1,
                'composition': composition,
                'predicted_ys': pred_ys,
                'predicted_uts': pred_uts,
                'predicted_elongation': pred_elong,
                'ys_error_percent': ys_error,
                'uts_error_percent': uts_error,
                'elongation_error_percent': elong_error,
                'fitness': fitness,
                'relative_cost': cost,
                'composition_string': format_composition_string(composition)
            })
        
        if verbose:
            print(f"\n✓ Optimization complete!")
            print(f"\nTop {top_k} Candidates:\n")
            
            for result in results:
                print(f"Rank {result['rank']}:")
                print(f"  Predicted: YS={result['predicted_ys']:.0f} MPa, "
                      f"UTS={result['predicted_uts']:.0f} MPa, "
                      f"Elong={result['predicted_elongation']:.1f}%")
                print(f"  Errors: YS±{result['ys_error_percent']:.1f}%, "
                      f"UTS±{result['uts_error_percent']:.1f}%, "
                      f"Elong±{result['elongation_error_percent']:.1f}%")
                print(f"  Relative Cost: {result['relative_cost']:.2f}x")
                print(f"  Fitness: {result['fitness']:.4f}")
                print()
        
        return results
    
    def optimize_with_constraints(self,
                                  target_ys: float,
                                  target_uts: float,
                                  target_elong: float,
                                  max_cost: Optional[float] = None,
                                  required_elements: Optional[Dict[str, Tuple[float, float]]] = None,
                                  **kwargs) -> List[Dict]:
        """
        Optimize with additional constraints
        
        Args:
            required_elements: Dict of element: (min, max) constraints
        """
        # Update bounds if required elements specified
        if required_elements:
            original_bounds = self.element_bounds.copy()
            for element, (min_val, max_val) in required_elements.items():
                if element in self.element_bounds:
                    self.element_bounds[element] = (min_val, max_val)
        
        # Add cost penalty if max_cost specified
        cost_penalty = kwargs.get('cost_penalty', 0.0)
        if max_cost:
            cost_penalty = max(cost_penalty, 0.5)  # Increase if cost limit
        
        results = self.optimize(target_ys, target_uts, target_elong, 
                              cost_penalty=cost_penalty, **kwargs)
        
        # Filter by cost if specified
        if max_cost:
            results = [r for r in results if r['relative_cost'] <= max_cost]
        
        # Restore original bounds
        if required_elements:
            self.element_bounds = original_bounds
        
        return results
