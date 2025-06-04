# api.py
from flask import Flask, jsonify, request
from src.data_processing import load_and_store_data, engineer_features, create_customer_segments
from src.ml_models import PredictiveModels
from src.optimization import PerformanceOptimizer
from flask import send_file
from src.advanced_visualization import AdvancedVisualization
from src.reporting import generate_csv_report, generate_excel_report

app = Flask(__name__)

# Load data once when starting the server
transactions_df, customers_df = load_and_store_data()
enhanced_df = engineer_features(transactions_df, customers_df)
segmented_df = create_customer_segments(enhanced_df)

@app.route('/api/segments', methods=['GET'])
def get_segments():
    segment_summary = segmented_df.groupby('segment').agg({
        'total_spend': ['mean', 'count'],
        'customer_lifetime_value': 'mean'
    }).round(2).to_dict()
    return jsonify(segment_summary)

@app.route('/api/customer/<customer_id>', methods=['GET'])
def get_customer_info(customer_id):
    customer_data = segmented_df[segmented_df['customer_id'] == customer_id].to_dict('records')
    if customer_data:
        return jsonify(customer_data[0])
    return jsonify({'error': 'Customer not found'}), 404
@app.route('/api/predict/segment', methods=['POST'])
def predict_segment():
    try:
        data = request.get_json()
        features = np.array(data['features'])
        model_type = data.get('model_type', 'ensemble')
        
        predictions = predictive_models.predict_customer_segment(
            features, model_type=model_type
        )
        
        return jsonify({
            'predictions': predictions.tolist(),
            'model_type': model_type
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/cohort/analysis', methods=['GET'])
def get_cohort_analysis():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    filtered_df = df
    if start_date and end_date:
        filtered_df = df[
            (df['transaction_date'] >= start_date) & 
            (df['transaction_date'] <= end_date)
        ]
    
    cohort_analysis = advanced_viz.create_cohort_analysis(filtered_df)
    return jsonify(cohort_analysis.to_dict())

@app.route('/api/export/analysis', methods=['GET'])
def export_analysis():
    format_type = request.args.get('format', 'csv')
    
    if format_type == 'csv':
        return send_file(
            generate_csv_report(),
            mimetype='text/csv',
            as_attachment=True,
            attachment_filename='analysis_report.csv'
        )
    elif format_type == 'excel':
        return send_file(
            generate_excel_report(),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            attachment_filename='analysis_report.xlsx'
        )

if __name__ == '__main__':
    app.run(debug=True)
