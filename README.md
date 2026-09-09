# Email Spam Detector

A beginner-friendly machine-learning project that classifies email text as **spam** or **not spam**.

## Technologies
- Python
- Pandas
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- Streamlit

## How to run

### 1. Open Terminal in this folder

### 2. Create a virtual environment (recommended)

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train.py
```

This creates:
`model/spam_model.pkl`

### 5. Start the web app
```bash
streamlit run app.py
```

A browser window will open. Paste an email and click **Check Email**.

## Example

Spam:
"Congratulations! You have won a $1000 gift card. Click here to claim now."

Normal:
"Please send me the project report before the meeting."

## Important
The included CSV is a small starter dataset for demonstration. For a stronger real-world detector, replace it with a larger, representative labeled email dataset and retrain the model.
