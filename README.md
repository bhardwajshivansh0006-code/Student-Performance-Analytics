# Student Performance Analytics and Prediction System

A Python-based summer training project that analyses sample student academic data and estimates a possible final score using a Random Forest Regression model.

## Project overview

The program uses these input features:
- Study hours per day
- Attendance percentage
- Assignment average
- Previous score

It analyses the data, trains a regression model, evaluates its predictions, and generates charts to help understand student performance.

## Technologies used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- CSV dataset

## Project files

- `student_performance_project.py` — main Python program
- `student_performance_sample.csv` — sample dataset used by the program
- `Shivansh_Kumar_Summer_Training_Project_Report.pdf` — project report

## How to run

1. Install Python 3.
2. Download or clone this repository.
3. Open a terminal in the project folder.
4. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

5. Run the program:

   ```bash
   python student_performance_project.py
   ```

Keep the Python file and CSV dataset in the same folder.

## Dataset note

The included 240-row dataset is synthetic sample data created for demonstration and learning. It does not contain real student records. The performance categories are illustrative, not official grading standards.

## Limitations

- The model is trained on synthetic sample data, so its results should not be treated as reliable real-world student predictions.
- It uses a limited set of academic features.
- Predicted scores are estimates and may differ from actual results.

## Future improvements

- Test with an authorised real-world dataset.
- Add a simple graphical user interface.
- Store records in a database.
- Add semester-wise analysis and model comparisons.

---

**Author:** Shivansh Kumar  
**Course:** B.Tech – CSE (Data Science)  
**Project type:** Python Programming Summer Training Project
