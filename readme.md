# Cat vs Dog Image Classification with SVM and HOG

This project demonstrates how to classify images of cats and dogs using Support Vector Machines (SVM) and Histogram of Oriented Gradients (H.O.G.) features. The workflow includes image preprocessing, feature extraction, model training, hyperparameter tuning, and evaluation.

## Requirements

- Python 3.x
- numpy
- opencv-python
- scikit-image
- scikit-learn
- joblib

Install dependencies with:

```bash
pip install numpy opencv-python scikit-image scikit-learn joblib
```

## Usage

1. Download the dataset and place images and `_annotations.json` in the `cats_dogs_images/` folder.
2. Run the Jupyter notebook `Image-classification-svm.ipynb`.
3. The notebook will:
    - Load images and labels
    - Extract H.O.G. features
    - Train and evaluate the SVM model
    - Save the best model as `svm_model.joblib`

## Project Workflow

1. **Image Preprocessing:**  
   Images are resized to 64x64 pixels and converted to grayscale.

2. **Feature Extraction:**  
   H.O.G. features are computed for each image.

3. **Label Encoding:**  
   Labels are mapped to integers (`cat` = 0, `dog` = 1).

4. **Data Splitting:**  
   The dataset is split into training and test sets with shuffling.

5. **Model Training:**  
   SVM is trained with hyperparameter tuning (`kernel`, `C`, `gamma`) using `GridSearchCV`.

6. **Evaluation:**  
   Model accuracy is printed for the test set.

7. **Model Saving:**  
   The best model is saved as `svm_model.joblib`.

## Conclusion

This project shows how classical machine learning techniques like SVM and H.O.G. can be effectively applied to image classification tasks. For further improvements, consider tuning H.O.G. parameters, balancing the dataset, or experimenting with more advanced algorithms.

## Author

Yasin Pourraisi  
- GitHub: [yasinpurraisi](https://github.com/yasinpurraisi)  
- Email: yasinpurraisi@gmail.com  
- Telegram:@yasinprsy