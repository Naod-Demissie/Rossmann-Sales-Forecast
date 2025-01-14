Here’s a `README.md` file explaining how to use the FastAPI-based prediction API located at `app/main.py`:

---

# Sales Prediction API

This API allows users to predict sales using a trained LSTM model. It preprocesses input data, makes predictions, and returns the predicted sales values.


## Running the API

To start the API, execute the following command:

```bash
python main.py
```

This will start the FastAPI server, accessible at `http://127.0.0.1:8000`.

## Testing the API

1. **Interactive Documentation**:
   Open your browser and navigate to:
   ```
   http://127.0.0.1:8000/docs
   ```
   Use the Swagger UI to interact with the API.

2. **Bash Command**:
   You can also test the API using `curl` or any HTTP client like `Postman`.

   Example `curl` command:
   ```bash
   curl -X POST "http://127.0.0.1:8000/predict/" \
   -H "Content-Type: application/json" \
   -d '{
       "Store": 1,
       "DayOfWeek": 5,
       "Date": "2022-09-15",
       "Sales": 0,
       "Customers": 55,
       "Open": 1,
       "Promo": 0,
       "StateHoliday": "0",
       "SchoolHoliday": 1
   }'
   ```

   Expected Response:
   ```json
   {
       "predicted_sales": 4567.89
   }
   ```

## API Endpoints

### `/predict/` (POST)
