Early-Warning System for Supply-Chain Disruption Risk

The objective of this project is to predict potential supply-chain disruptions using historical trade, logistics, and risk indicators, enabling proactive decision-making.

Dataset Selection and Official Sources 

1.	World Bank – Logistics Performance Index (LPI)
		Used to analyze logistics efficiency, infrastructure quality, and shipment reliability.
2.	World Bank – International Trade and Economic Indicators
		Includes import/export volumes, GDP, inflation, and trade dependency metrics.
3.	Uppsala Conflict Data Program (UCDP)
		Provides structured data on regional instability and conflict events that can impact supply chains.
4.	SIPRI Military Expenditure Database
		Used as an indicator of geopolitical stress and regional risk.

These datasets collectively represent economic, logistical, and geopolitical factors affecting global supply chains.



6. Dataset Analysis

The selected datasets were analyzed to understand their structure and applicability:
	•	The datasets contain country-level and time-based numerical attributes.
	•	Data is collected from multiple independent sources and requires integration.
	•	Some datasets contain missing values due to reporting gaps.
	•	Feature values exist at different scales and ranges.
	•	The datasets are suitable for classification, regression, and clustering tasks after preprocessing.



7. Data Preparation Techniques Applied

The following data preparation techniques were applied to prepare the data for machine learning applications:

1. Data Integration

Multiple datasets from different sources were merged based on common attributes such as country and year.

2. Handling Missing Values

Missing values were identified and handled using statistical imputation techniques to maintain data consistency.

3. Feature Selection

Relevant indicators contributing to supply-chain disruption risk were selected to reduce redundancy and noise.

4. Feature Scaling

Numerical features were normalized to bring all attributes to a comparable scale.

5. Outlier Detection

Extreme values were analyzed to prevent skewed learning and incorrect predictions.

6. Train-Test Split

The dataset was divided into training and testing subsets for evaluation of machine learning models.

post this the model would be trained on various domains , we find the accuracy , precision , recall and F1 score of each for comparison



