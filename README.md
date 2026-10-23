# task4-data-dashboard-with-ai
This project is a data analytics application built with Python and Streamlit. It was created as part of Task #4 during my data engineering training.

The application works with data from five stores. It processes data about users, books, and orders, and presents the results in an interactive dashboard.

Project structure:
    - app.py - main application
    - src/ - data processing, analytics, dashboard control, AI Chat
    - analyze_data/ - scripts for analyzing raw data
    - data/ - input data
    - output/ - generated charts
    - run_all.py - script for running data analysis tasks

Running the application
    1. Clone this repository
    2. Install required packages -> 'pip install -r requirements.txt'
    3. Configure the GROQ_API_KEY secret for the AI chat. The key is already configured for the online application, so no additional setup is needed when using it in a browser.
    4. Start the application -> 'streamlit run app.py'

    * If you change some data in /data folder just run -> 'python run_all.py' before starting the application.

This project was developed as a learning exercise to practice Python, data analysis, and building interactive data applications.