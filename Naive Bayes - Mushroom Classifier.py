import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Feature mappings
feature_mappings = {
    'CAPSHAPE': ['convex', 'bell', 'sunken', 'flat', 'knobbed', 'conical'],
    'SURFACE': ['smooth', 'scaly', 'fibrous', 'grooves'],
    'COLOR': ['brown', 'yellow', 'white', 'gray', 'red', 'pink', 'buff', 'purple', 'cinnamo', 'green'],
    'BRUISES': ['no', 'yes'],
    'ODOR': ['pungent', 'almond', 'anise', 'none', 'foul', 'creosote', 'fishy', 'spicy', 'musty'],
    'GILL-ATTACHMENT': ['free', 'attached'],
    'GILL-SPACING': ['close', 'crowded'],
    'GILL-SIZE': ['narrow', 'broad'],
    'GILL-COLOR': ['black', 'brown', 'gray', 'pink', "b'w'", 'chocolate', 'purple', 'red', 'buff', 'green', "b'y'", 'orange'],
    'STALK-SHAPE': ['enlarging', 'tapering'],
    'STALK-ROOT': ['equal', 'club', 'bulbous', 'rooted', 'missing'],
    'STALK-SURFACE-ABOVE-RING': ['smooth', 'fibrous', 'silky', 'scaly'],
    'STALK-SURFACE-BELOW-RING': ['smooth', 'fibrous', 'silky', 'scaly'],
    'STALK-COLOR-ABOVE-RING': ['brown', 'yellow', 'white', 'gray', 'red', 'pink', 'buff', 'orange', 'cinnamo'],
    'STALK-COLOR-BELOW-RING': ['brown', 'yellow', 'white', 'gray', 'red', 'pink', 'buff', 'orange', 'cinnamo'],
    'VEIL-TYPE': ['partial'],
    'VEIL-COLOR': ['white', 'brown', 'orange', 'yellow'],
    'RING-NUMBER': ['one', 'two', 'none'],
    'RING-TYPE': ['pendant', 'evanescent', 'large', 'flaring', 'none'],
    'SPORE-PRINT-COLOR': ['black', 'brown', 'purple', 'chocolate', 'white', 'green', 'orange', 'yellow', 'buff'],
    'POPULATION': ['scattered', 'numerous', 'abundant', 'several', 'solitary', 'clustered'],
    'HABITAT': ['urban', 'grasses', 'meadows', 'wood', 'path', 'waste', 'leaves']
}

# Load the dataset
data = pd.read_csv("MushroomCSV.csv")

# Separate features and target variable
X = data.drop('CLASS', axis=1)
y = data['CLASS']

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X)

# Train the RandomForestClassifier
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent value
    ('classifier', RandomForestClassifier())
])
pipeline.fit(X, y)

# Create GUI
window = tk.Tk()
window.title("Mushroom Classifier")

# Function to classify mushroom
def classify_mushroom():
    # Create an empty DataFrame with columns matching the training data
    test_data = pd.DataFrame(columns=X.columns)
    
    # Populate the test data DataFrame with the user inputs
    for feature, options in feature_mappings.items():
        selected_option = feature_entries[feature].get()
        # One-hot encode the selected option
        one_hot_encoded = pd.get_dummies([selected_option], prefix=feature)
        # Add only the relevant columns to the test data DataFrame
        for column in one_hot_encoded.columns:
            if column in test_data.columns:
                test_data[column] = one_hot_encoded[column]
    
    # Predict the class of the mushroom
    prediction = pipeline.predict(test_data)
    probability = pipeline.predict_proba(test_data)[0]
    
    # Show message box with result
    if prediction[0] == 'edible':
        result = "Edible"
    else:
        result = "Poisonous"
    messagebox.showinfo("Prediction Result", f"This mushroom is {result} with probability {probability}")

    # Display figure
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.bar(['Edible', 'Poisonous'], probability, color=['yellow', 'purple'])
    ax.set_ylabel('Probability')
    ax.set_title('Probability of Edibility')
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=2, rowspan=100)

# Create input fields for each feature
feature_entries = {}
row_num = 0
for feature, options in feature_mappings.items():
    feature_label = tk.Label(window, text=feature + ":")
    feature_label.grid(row=row_num, column=0, sticky="w")
    feature_entries[feature] = tk.StringVar(window)
    feature_entries[feature].set(options[0])  # Default value
    feature_combobox = ttk.Combobox(window, textvariable=feature_entries[feature], values=options, state="readonly")
    feature_combobox.grid(row=row_num, column=1, sticky="w")
    if row_num % 2 != 0:
        row_num += 1
    else:
        row_num += 1

classify_button = tk.Button(window, text="Classify Mushroom", command=classify_mushroom)
classify_button.grid(row=row_num, column=0, columnspan=2)

window.mainloop()
