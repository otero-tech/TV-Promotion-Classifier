# TV Promotion Email Click Prediction

## Project Overview

This project predicts which users are most likely to click on a promotional email for a TV sale. The goal is to target only the users who are genuinely interested, improving the effectiveness of the marketing campaign while reducing email fatigue among uninterested users.

The prediction is based on historical data of user behavior, including browsing history and past interactions with promotional emails. A machine learning classifier analyzes user data to predict future engagement.

## Data

The project uses three primary datasets:

- **User Data** (`train_users.csv`, `test1_users.csv`): Information about individual users, such as their ID and past purchase amounts.
- **Log Data** (`train_logs.csv`, `test1_logs.csv`): Details of each user's visits to different webpages, including the date, URL, and duration of each visit.
- **Clicked Data** (`train_clicked.csv`, `test1_clicked.csv`): Indicates whether a user clicked on a previous promotional email (`clicked=1` for clicked, `clicked=0` for not clicked).

## UserPredictor Class

The core of the project is the `UserPredictor` class, which implements a machine learning model to predict user engagement with promotional emails. 

### Key Methods

1. **`fit(X_users, X_logs, y)`**: 
   - Trains the model using user data (`X_users`), log data (`X_logs`), and the target variable (`y`), which indicates whether the user clicked the email.

2. **`predict(X_users, X_logs)`**:
   - Predicts whether users in the provided datasets (`X_users`, `X_logs`) are likely to click on the promotional email.

### Feature Engineering

- **User Features**: Extracts features such as `id` and `past_purchase_amt` from user data.
- **Log Features**: Processes user browsing history, including time spent on the website, number of page visits, and types of products viewed.
- **URL-Based Features**: Utilizes one-hot encoding for product types based on URL patterns to capture user interests.

### Model

The `UserPredictor` class uses a pipeline with:

- **StandardScaler**: Normalizes the data to improve model performance.
- **LogisticRegression**: Applies L1 regularization to predict the likelihood of a user clicking on the promotional email.

## Results

The model predicts whether a user is likely to click on a promotional email based on their historical behavior. Performance can be optimized by adjusting features and experimenting with different machine learning techniques.

