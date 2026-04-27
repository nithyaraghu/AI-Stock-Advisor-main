# AI Financial Advisor

This repository contains the backend code for the AI Agent application, built with Flask. Follow the steps below to set up and run the project locally.

## Prerequisites

- Python 3.8 or above
- Git
- Visual Studio Code (or any other IDE with terminal support)

## Installation Instructions

### Step 1: Clone the Repository

Clone the repository to your local machine using the following command in your terminal: 
``` git clone https://github.com/asish-kun/ai-financial-advisor.git ```

### Step 2: Open the Project

Open the folder `ai-agent-bdr-backend` in Visual Studio Code or your preferred IDE.

### Step 3: Set Up the Python Environment

Open a terminal in your IDE and run the following commands to set up a virtual environment and install the required dependencies:

1. Create a virtual environment:
``` python -m venv venv ```

2. Activate the virtual environment:
- **Windows**: ``` venv\Scripts\activate ```
- **macOS/Linux**: ``` source venv/bin/activate ```

3. Install the required Python packages:
``` pip install -r requirements.txt ```

### Step 4: Environment Variables

Create a `.env` file in the root directory of the project. Populate it with the necessary key-value pairs.

### Step 5: Run the Application

Execute the following command to start the server: ``` python app.py ```

You should see logs in the console indicating that the server is running along with any other output or debug information.
