#!/usr/bin/env python3
"""
Improved Analysis Runner Script
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from ae_bayesian.analysis.improved_analysis import ImprovedPriorComparisonAnalysis

def main():
    """Main execution function"""
    print("=" * 60)
    print("Improved Detailed Analysis")
    print("=" * 60)
    
    # Create analysis instance
    analyzer = ImprovedPriorComparisonAnalysis()
    
    # Execute analysis
    summary_stats = analyzer.run_comprehensive_analysis()
    
    print("\nAnalysis completed! Results saved in results directory.")
    
    return summary_stats

if __name__ == "__main__":
    main()
