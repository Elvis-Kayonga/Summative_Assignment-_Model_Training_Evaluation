# Breast Cancer Wisconsin: Traditional ML vs Deep Learning Comparative Analysis

## 📋 Project Overview

This project implements a comprehensive comparative study between traditional machine learning and deep learning approaches for breast cancer diagnosis using the Wisconsin Diagnostic Breast Cancer dataset from the UCI Machine Learning Repository.

**Domain:** Healthcare - Oncology  
**Task:** Binary Classification (Malignant vs Benign)  
**Dataset:** 569 samples, 30 features  
**Date:** February 19, 2026

## 🎯 Objectives

- Compare traditional ML (Scikit-learn) with deep learning (TensorFlow) approaches
- Conduct 10+ structured experiments with systematic hyperparameter exploration
- Demonstrate production-ready implementation patterns (tf.data, checkpointing)
- Provide comprehensive error analysis and theoretical grounding
- Ensure full reproducibility with proper data versioning

## 📊 Experiments Summary

### Traditional Machine Learning (Experiments 1-4)
1. **EXP-01:** Logistic Regression (Baseline)
2. **EXP-02:** Logistic Regression with L1/L2 Regularization
3. **EXP-03:** Random Forest Classifier with Feature Importance
4. **EXP-04:** Support Vector Machine (Linear vs RBF Kernels)

### Deep Learning (Experiments 5-10)
5. **EXP-05:** Basic Sequential Neural Network
6. **EXP-06:** Sequential NN with Dropout Regularization
7. **EXP-07:** Sequential NN with L2 Weight Regularization
8. **EXP-08:** Functional API with Complex Architecture
9. **EXP-09:** tf.data Pipeline Implementation
10. **EXP-10:** Learning Rate Comparison (0.01, 0.001, 0.0001)

## 📁 Project Structure

```
Summative_Assignment-_Model_Training_Evaluation/
│
├── breast_cancer_ml_dl_comparison.ipynb    # Main notebook
├── README.md                                # This file
│
├── data/                                    # Data files
│   ├── breast_cancer_preprocessed.csv
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   └── scaler.pkl
│
├── models/                                  # Saved models
│   ├── exp1_logistic_regression_baseline.pkl
│   ├── exp2a_logistic_regression_l1.pkl
│   ├── exp2b_logistic_regression_l2.pkl
│   ├── exp3_random_forest.pkl
│   ├── exp4a_svm_linear.pkl
│   ├── exp4b_svm_rbf.pkl
│   ├── exp5_basic_sequential.h5
│   ├── exp6_sequential_dropout.h5
│   ├── exp7_sequential_l2.h5
│   ├── exp8_functional_api.h5
│   ├── exp9_tfdata_pipeline.h5
│   └── exp10_lr_*.h5
│
├── figures/                                 # Visualizations
│   ├── class_distribution.png
│   ├── correlation_matrix.png
│   ├── feature_importance.png
│   ├── exp1_evaluation.png
│   ├── exp*_evaluation.png
│   ├── exp*_learning_curves.png
│   ├── exp10_learning_rate_comparison.png
│   └── final_performance_comparison.png
│
└── results/                                 # Experiment tracking
    └── experiment_results.csv
```

## 🔧 Installation and Setup

### Requirements

```bash
# Python 3.8+
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install scikit-learn==1.3.0
pip install tensorflow==2.15.0
pip install ucimlrepo==0.0.3
pip install joblib
```

### Running the Notebook

1. **Clone/Download** this repository
2. **Install dependencies** using the command above
3. **Open Jupyter Notebook or JupyterLab:**
   ```bash
   jupyter notebook breast_cancer_ml_dl_comparison.ipynb
   ```
4. **Run all cells** from top to bottom (Cell → Run All)

### Reproducibility

- All random seeds are set (numpy, tensorflow, sklearn)
- Deterministic operations configured
- Data splits and preprocessing are saved and versioned
- Results logged incrementally to CSV

## 📈 Key Features

### Academic Rigor
- ✅ Theoretical grounding for each experiment
- ✅ Bias-variance trade-off analysis
- ✅ Learning curve interpretation
- ✅ Confusion matrix and ROC curve analysis
- ✅ Clinical implications discussion
- ✅ Dataset limitations critique

### Production Readiness
- ✅ ModelCheckpoint callbacks
- ✅ EarlyStopping for efficient training
- ✅ tf.data pipeline with prefetching
- ✅ Crash recovery with checkpointing
- ✅ Comprehensive logging

### Visualization Quality
- ✅ Professional matplotlib/seaborn plots
- ✅ All plots saved to `figures/` directory
- ✅ Titles, labels, legends, grids
- ✅ High-resolution (300 DPI) outputs

### Code Quality
- ✅ Modular, reusable functions
- ✅ Clear markdown explanations
- ✅ Proper error handling
- ✅ Comprehensive documentation

## 📊 Results Highlights

All models achieved >95% accuracy on the test set, with minimal performance differences between traditional ML and deep learning approaches. Key findings:

- **Best Traditional ML:** Random Forest and SVM (RBF kernel)
- **Best Deep Learning:** Sequential NN with Dropout
- **Most Interpretable:** Logistic Regression with L2
- **Production Ready:** tf.data pipeline implementation

**Conclusion:** For this dataset size (569 samples), traditional ML and DL perform comparably. Traditional ML offers interpretability advantages; DL provides flexibility for scaling to larger datasets or complex data types.

## 🎓 Academic Use

This notebook is designed to meet academic assessment criteria:

- **Problem Definition:** Original analysis with clinical context
- **Literature Grounding:** References to ML theory and medical AI
- **Methodology:** Rigorous preprocessing, feature engineering, model implementation
- **Experimentation:** 10+ experiments with systematic variation
- **Evaluation:** Comprehensive metrics, learning curves, error analysis
- **Critical Reflection:** Dataset limitations and generalization concerns
- **Reproducibility:** Fully documented and deterministic

### Report Integration

The notebook outputs are designed for integration into an academic report:
- Tables can be exported to LaTeX or Word
- All figures are high-resolution and publication-ready
- Analysis sections provide narrative for discussion
- Experiment results CSV enables custom visualizations

## 📚 References

1. Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995). Image analysis and machine learning applied to breast cancer diagnosis and prognosis. *Analytical and Quantitative Cytology and Histology*, 17(2), 77-87.

2. UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic) Dataset. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

4. Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*. Springer.

## 🤝 Contributing

This is an academic project. For questions or suggestions:
- Open an issue on GitHub
- Contact: [Your Email]

## 📄 License

[Specify your license here - e.g., MIT, Apache 2.0, or Academic Use Only]

## ⚠️ Disclaimer

This project is for educational and research purposes only. It is **not** intended for clinical use. Any medical application would require:
- Regulatory approval (FDA, CE marking, etc.)
- External validation on independent datasets
- Prospective clinical trials
- Integration with clinical workflows
- Continuous monitoring and recalibration

## 🙏 Acknowledgments

- UCI Machine Learning Repository for dataset access
- Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian for dataset creation
- TensorFlow and Scikit-learn communities for excellent documentation
- Academic instructors for project guidance

---

**Author:** [Your Name]  
**Institution:** [Your Institution]  
**Date:** February 19, 2026  
**Course:** [Course Name/Code]

---

*This project demonstrates best practices in machine learning experimentation, model evaluation, and academic research methodology.*
