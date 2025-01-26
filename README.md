<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Urban Heat Index Prediction and Mitigation Tool</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        code {
            background: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
        }
        a {
            color: #3498db;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .section {
            margin-bottom: 30px;
        }
        .section h2 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }
    </style>
</head>
<body>

    <h1>Urban Heat Index Prediction and Mitigation Tool 🌡️</h1>

    <div class="section">
        <h2>Overview</h2>
        <p>
            This project is a <strong>Streamlit-based web application</strong> designed to predict urban heat island effects and provide actionable mitigation strategies. It uses machine learning (Random Forest Regressor) to predict land surface temperature based on various urban and environmental features. The app also includes <strong>Explainable AI (XAI)</strong> visualizations and <strong>heat mitigation recommendations</strong> based on user inputs.
        </p>
    </div>

    <div class="section">
        <h2>Features</h2>
        <ul>
            <li><strong>Temperature Prediction</strong>: Predicts land surface temperature based on urban and environmental features.</li>
            <li><strong>Explainable AI (XAI)</strong>: Displays feature importance to explain model predictions.</li>
            <li><strong>Heat Mitigation Recommendations</strong>: Provides actionable suggestions to reduce urban heat island effects.</li>
            <li><strong>Interactive Interface</strong>: User-friendly input fields for urban parameters.</li>
            <li><strong>Scalable and Customizable</strong>: Easily adaptable to different cities and datasets.</li>
        </ul>
    </div>

    <div class="section">
        <h2>Installation</h2>
        <p>To run this project locally, follow these steps:</p>
        <h3>Prerequisites</h3>
        <ul>
            <li>Python 3.9 or higher</li>
            <li>Streamlit</li>
            <li>Scikit-learn</li>
            <li>Pandas</li>
            <li>Joblib</li>
            <li>Pillow</li>
        </ul>
        <h3>Steps</h3>
        <ol>
            <li>Clone the repository:
                <pre><code>git clone https://github.com/your-username/urban-heat-index.git
cd urban-heat-index</code></pre>
            </li>
            <li>Install dependencies:
                <pre><code>pip install -r requirements.txt</code></pre>
            </li>
            <li>Run the Streamlit app:
                <pre><code>streamlit run app.py</code></pre>
            </li>
            <li>Open your browser and navigate to <a href="http://localhost:8501" target="_blank">http://localhost:8501</a>.</li>
        </ol>
    </div>

    <div class="section">
        <h2>Usage</h2>
        <ol>
            <li><strong>Input Urban Parameters</strong>: Use the sidebar to input values for features like latitude, longitude, green cover percentage, building height, etc.</li>
            <li><strong>Predict Temperature</strong>: Click the "Analyze Urban Heat" button to get the predicted land surface temperature.</li>
            <li><strong>View Recommendations</strong>: The app will display heat mitigation strategies based on the input parameters and predicted temperature.</li>
            <li><strong>Explore Feature Importance</strong>: Check the XAI visualization to understand which features contribute most to the prediction.</li>
        </ol>
    </div>

    <div class="section">
        <h2>Dataset</h2>
        <p>The model is trained on a dataset containing the following features:</p>
        <ul>
            <li><strong>Geospatial Features</strong>: Latitude, Longitude</li>
            <li><strong>Environmental Features</strong>: Green Cover Percentage, Albedo, Urban Vegetation Index, Proximity to Water Body</li>
            <li><strong>Urban Infrastructure</strong>: Building Height, Road Density, Surface Material</li>
            <li><strong>Climate Factors</strong>: Relative Humidity, Wind Speed, Solar Radiation, Nighttime Surface Temperature</li>
            <li><strong>Population & Emissions</strong>: Population Density, Carbon Emission Levels</li>
            <li><strong>Urban Connectivity</strong>: Distance from Previous Point</li>
        </ul>
    </div>

    <div class="section">
        <h2>Model Training</h2>
        <p>The Random Forest Regressor model is trained using the following steps:</p>
        <ol>
            <li><strong>Data Preprocessing</strong>: Handle missing values (if any) and encode categorical features (e.g., Surface Material).</li>
            <li><strong>Model Training</strong>: Split the dataset into training and testing sets, and train the model using <code>RandomForestRegressor</code> from Scikit-learn.</li>
            <li><strong>Model Saving</strong>: Save the trained model as a <code>.pkl</code> file using <code>joblib</code>.</li>
        </ol>
    </div>

    <div class="section">
        <h2>Deployment</h2>
        <p>The app can be deployed on <strong>Streamlit Sharing</strong> or any other cloud platform. Follow these steps:</p>
        <ol>
            <li>Push your code to a GitHub repository.</li>
            <li>Go to <a href="https://share.streamlit.io/" target="_blank">Streamlit Sharing</a>.</li>
            <li>Connect your GitHub repository.</li>
            <li>Deploy the app by specifying the path to <code>app.py</code>.</li>
        </ol>
    </div>

    <div class="section">
        <h2>Customization</h2>
        <ul>
            <li><strong>Heat Thresholds</strong>: Modify the <code>HEAT_THRESHOLDS</code> dictionary in <code>app.py</code> to adjust thresholds for recommendations.</li>
            <li><strong>Model</strong>: Replace <code>trainedmodel.pkl</code> with your own trained model.</li>
            <li><strong>Dataset</strong>: Update the dataset URL in the training script to use a different dataset.</li>
        </ul>
    </div>

    <div class="section">
        <h2>Contributing</h2>
        <p>Contributions are welcome! If you'd like to contribute, please follow these steps:</p>
        <ol>
            <li>Fork the repository.</li>
            <li>Create a new branch (<code>git checkout -b feature/YourFeatureName</code>).</li>
            <li>Commit your changes (<code>git commit -m 'Add some feature'</code>).</li>
            <li>Push to the branch (<code>git push origin feature/YourFeatureName</code>).</li>
            <li>Open a pull request.</li>
        </ol>
    </div>

    <div class="section">
        <h2>License</h2>
        <p>This project is licensed under the MIT License. See the <a href="LICENSE" target="_blank">LICENSE</a> file for details.</p>
    </div>

    <div class="section">
        <h2>Acknowledgments</h2>
        <ul>
            <li><strong>Scikit-learn</strong>: For providing the machine learning framework.</li>
            <li><strong>Streamlit</strong>: For enabling the interactive web app.</li>
            <li><strong>Open Data Providers</strong>: For making urban heat datasets publicly available.</li>
        </ul>
    </div>

    <div class="section">
        <h2>Contact</h2>
        <p>For questions or feedback, please contact:</p>
        <ul>
            <li><strong>Your Name</strong>: <a href="mailto:your.email@example.com">your.email@example.com</a></li>
            <li><strong>GitHub</strong>: <a href="https://github.com/your-username" target="_blank">your-username</a></li>
        </ul>
    </div>

</body>
</html>
