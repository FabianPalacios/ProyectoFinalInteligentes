import os
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
from sklearn.decomposition import PCA
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
import os

app = Flask(__name__)

# Configuración de la base de datos MongoDB
mongo_uri = 'mongodb+srv://Fabian:21jejAlo@proyectofinal.gaj5fmt.mongodb.net/?retryWrites=true&w=majority'
client = MongoClient(mongo_uri)
db = client['InteligentesDB']
collection = db['Inteligentes']

# Function to convert CSV to JSON with a specified delimiter
def csv_to_json(file_path, delimiter=';'):
    df = pd.read_csv(file_path, delimiter=delimiter)
    json_data = df.to_json(orient='records')
    return json_data

# Function to convert Excel to JSON with a specified delimiter
def excel_to_json(file_path, delimiter=';'):
    df = pd.read_excel(file_path)
    json_data = df.to_json(orient='records')
    return json_data

# Load data endpoint
@app.route('/load', methods=['POST'])
def load_data():
    try:
        # Check if a file was sent
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó ningún archivo'}), 400

        file = request.files['file']

        # Check if the file has an allowed extension
        allowed_extensions = {'xlsx', 'xls', 'csv'}
        if file.filename.split('.')[-1] not in allowed_extensions:
            return jsonify({'error': 'Formato de archivo no permitido'}), 400

        # Save the file temporarily
        file_path = 'temp_file.' + file.filename.split('.')[-1]
        file.save(file_path)

        # Convert the file to JSON based on its type
        if file.filename.endswith('.csv'):
            json_data = csv_to_json(file_path, delimiter=';')
        else:
            json_data = excel_to_json(file_path)

        # Modify the JSON structure to include the "LoadExcel" tag
        final_data = {"LoadExcel": pd.read_json(json_data, orient='records').to_dict(orient='records')}

        # Store the data in MongoDB and get the _id
        result = collection.insert_one(final_data)
        inserted_id = str(result.inserted_id)

        # Remove the temporary file
        os.remove(file_path)

        # Return the _id in the response
        return jsonify({'message': 'Datos cargados correctamente', '_id': inserted_id}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/basic statistics/<string:dataset_id>', methods=['GET'])
def get_basic_statistics(dataset_id):
    try:
        # Convert the dataset_id to ObjectId
        object_id = ObjectId(dataset_id)

        # Find the document in the collection
        document = collection.find_one({"_id": object_id})

        if not document:
            return jsonify({'error': 'Dataset no encontrado'}), 404

        # Convert the data to a DataFrame
        df = pd.DataFrame(document['LoadExcel'])

        # Generate basic statistics using describe for all columns
        statistics = df.describe(include='all').to_dict()

        return jsonify(statistics), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/columns-describe/<string:dataset_id>', methods=['GET'])
def get_column_data_types(dataset_id):
    try:
        # Remove leading and trailing spaces and convert the dataset_id to ObjectId
        object_id = ObjectId(dataset_id.strip())

        # Find the document in the collection
        document = collection.find_one({"_id": object_id})

        if not document:
            return jsonify({'error': 'Dataset no encontrado'}), 404

        # Convert the data to a DataFrame
        df = pd.DataFrame(document['LoadExcel'])

        # Identify data types of each column
        column_data_types = df.dtypes.apply(lambda x: 'Numeric' if pd.api.types.is_numeric_dtype(x) else 'Text').to_dict()

        return jsonify(column_data_types), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/imputation/<string:dataset_id>/type/<string:imputation_type>', methods=['POST'])
def impute_missing_data(dataset_id, imputation_type):
    try:
        # Remove leading and trailing spaces and convert the dataset_id to ObjectId
        object_id = ObjectId(dataset_id.strip())

        # Find the document in the collection
        document = collection.find_one({"_id": object_id})

        if not document:
            return jsonify({'error': 'Dataset no encontrado'}), 404

        # Convert the data to a DataFrame
        original_df = pd.DataFrame(document['LoadExcel'])

        # Create a copy of the original dataset
        imputed_df = original_df.copy()

        # Check the imputation type
        if imputation_type == '1':
            # Type 1: Remove rows with missing data
            imputed_df = imputed_df.dropna()

        elif imputation_type == '2':
            # Type 2: Impute missing values based on type
            for column in imputed_df.columns:
                if pd.api.types.is_numeric_dtype(imputed_df[column]):
                    # Impute numeric variables with mean
                    imputed_df[column].fillna(imputed_df[column].mean(), inplace=True)
                else:
                    # Impute categorical/text variables with mode
                    imputed_df[column].fillna(imputed_df[column].mode()[0], inplace=True)

        else:
            return jsonify({'error': 'Tipo de imputación no válido'}), 400

        # Update the existing document with the imputed data
        collection.update_one({"_id": object_id}, {"$set": {"ImputedData": imputed_df.to_dict(orient='records')}})

        return jsonify({'message': 'Imputación realizada correctamente', 'imputed_dataset_id': str(object_id)}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/general-univariate-graphs/<string:dataset_id>', methods=['POST'])
def generate_univariate_graphs(dataset_id):
    try:
        # Remove leading and trailing spaces and convert the dataset_id to ObjectId
        object_id = ObjectId(dataset_id.strip())

        # Find the document in the collection
        document = collection.find_one({"_id": object_id})

        if not document:
            return jsonify({'error': 'Dataset no encontrado'}), 404

        # Convert the data to a DataFrame
        df = pd.DataFrame(document['ImputedData'])

        # Create a main folder with the dataset_id as the folder name
        main_folder_path = f"graphs/{dataset_id}"
        os.makedirs(main_folder_path, exist_ok=True)

        # Create subfolders for each type of graph
        histogram_folder = os.path.join(main_folder_path, "histograms")
        os.makedirs(histogram_folder, exist_ok=True)

        boxplot_folder = os.path.join(main_folder_path, "boxplots")
        os.makedirs(boxplot_folder, exist_ok=True)

        distribution_folder = os.path.join(main_folder_path, "distributions")
        os.makedirs(distribution_folder, exist_ok=True)

        # Generate and save univariate graphs for each column
        for column in df.columns:
            # Histogram
            plt.figure(figsize=(10, 6))
            sns.histplot(df[column], kde=False, bins=20)
            plt.title(f'Histogram for {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            histogram_path = os.path.join(histogram_folder, f"{column}_histogram.png")
            plt.savefig(histogram_path)
            plt.close()

            # Box plot (for numeric variables only)
            if pd.api.types.is_numeric_dtype(df[column]):
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=df[column])
                plt.title(f'Box Plot for {column}')
                plt.xlabel(column)
                plt.ylabel('Values')
                boxplot_path = os.path.join(boxplot_folder, f"{column}_boxplot.png")
                plt.savefig(boxplot_path)
                plt.close()

            # Probability distribution plot
            if pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_datetime64_any_dtype(df[column]):
                plt.figure(figsize=(10, 6))
                sns.kdeplot(df[column], fill=True)
                plt.title(f'Probability Distribution for {column}')
                plt.xlabel(column)
                plt.ylabel('Density')
                distribution_path = os.path.join(distribution_folder, f"{column}_distribution.png")
                plt.savefig(distribution_path)
                plt.close()

        return jsonify({'message': 'Gráficos generados correctamente', 'main_folder': main_folder_path}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/univariate-graphs-class/<string:dataset_id>', methods=['POST'])
def generate_univariate_graphs_class(dataset_id):
    try:
        # Remove leading and trailing spaces and convert the dataset_id to ObjectId
        object_id = ObjectId(dataset_id.strip())

        # Find the document in the collection
        document = collection.find_one({"_id": object_id})

        if not document:
            return jsonify({'error': 'Dataset no encontrado'}), 404

        # Convert the data to a DataFrame
        df = pd.DataFrame(document['ImputedData'])

        # Check if the 'class' column exists in the dataset
        if 'class' not in df.columns:
            return jsonify({'error': 'No se encontró la columna "class" en el dataset'}), 400

        # Create a main folder with the dataset_id as the folder name
        main_folder_path = f"graphs_class/{dataset_id}"
        os.makedirs(main_folder_path, exist_ok=True)

        # Create subfolders for each type of graph
        boxplot_folder = os.path.join(main_folder_path, "boxplots_class")
        os.makedirs(boxplot_folder, exist_ok=True)

        density_folder = os.path.join(main_folder_path, "density_class")
        os.makedirs(density_folder, exist_ok=True)

        # Generate and save univariate graphs for each column
        for column in df.columns:
            # Box plot per class (for numeric variables only)
            if pd.api.types.is_numeric_dtype(df[column]):
                plt.figure(figsize=(12, 8))
                sns.boxplot(x='class', y=column, data=df)
                plt.title(f'Box Plot for {column} by Class')
                plt.xlabel('Class')
                plt.ylabel(column)
                boxplot_path = os.path.join(boxplot_folder, f"{column}_boxplot_class.png")
                plt.savefig(boxplot_path)
                plt.close()

            # Density plot per class (for numeric variables only)
            if pd.api.types.is_numeric_dtype(df[column]):
                plt.figure(figsize=(12, 8))
                for class_value in df['class'].unique():
                    sns.kdeplot(df[df['class'] == class_value][column], label=f'Class {class_value}', fill=True)
                plt.title(f'Density Plot for {column} by Class')
                plt.xlabel(column)
                plt.ylabel('Density')
                density_path = os.path.join(density_folder, f"{column}_density_class.png")
                plt.legend()
                plt.savefig(density_path)
                plt.close()

        return jsonify({'message': 'Gráficos generados correctamente', 'main_folder': main_folder_path}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/bivariate-graphs-class/<string:dataset_id>', methods=['GET'])
def generate_bivariate_graphs_class(dataset_id):
    try:
        # Remove leading and trailing spaces and convert the dataset_id to ObjectId
        object_id = ObjectId(dataset_id.strip())

        # Find the document in the collection
        document = collection.find_one({"_id": object_id})

        if not document:
            return jsonify({'error': 'Dataset no encontrado'}), 404

        # Convert the data to a DataFrame
        df = pd.DataFrame(document['ImputedData'])

        # Check if the 'class' column exists in the dataset
        if 'class' not in df.columns:
            return jsonify({'error': 'No se encontró la columna "class" en el dataset'}), 400

        # Group by class and create pair plot
        plt.figure(figsize=(15, 15))
        sns.pairplot(df, hue='class', palette='Set1', markers=["o", "s", "D"])
        plt.suptitle("Pair Plot by Class", y=1.02)
        
        # Save the pair plot image
        pairplot_image_path = f"pairplot_class_{dataset_id}.png"
        plt.savefig(pairplot_image_path)
        plt.close()

        # Return the link to the saved pair plot image
        pairplot_image_link = f"http://tu_server/{pairplot_image_path}"

        return jsonify({'pairplot_link': pairplot_image_link}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/multivariate-graphs-class/<string:dataset_id>', methods=['GET'])
def generate_multivariate_graphs_class(dataset_id):
    try:
        # Remove leading and trailing spaces and convert the dataset_id to ObjectId
        object_id = ObjectId(dataset_id.strip())

        # Find the document in the collection
        document = collection.find_one({"_id": object_id})

        if not document:
            return jsonify({'error': 'Dataset no encontrado'}), 404

        # Convert the data to a DataFrame
        df = pd.DataFrame(document['ImputedData'])

        # Check if the dataset has numeric columns for correlation analysis
        numeric_columns = df.select_dtypes(include=['number']).columns
        if len(numeric_columns) == 0:
            return jsonify({'error': 'El dataset no contiene columnas numéricas para el análisis de correlación'}), 400

        # Calculate the correlation matrix
        correlation_matrix = df[numeric_columns].corr()

        # Plot the correlation matrix heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title("Correlation Matrix of Numeric Columns")
        
        # Save the correlation matrix plot image
        correlation_matrix_image_path = f"correlation_matrix_{dataset_id}.png"
        plt.savefig(correlation_matrix_image_path)
        plt.close()

        # Return the link to the saved correlation matrix plot image
        correlation_matrix_image_link = f"http://tu_server/{correlation_matrix_image_path}"

        return jsonify({'correlation_matrix_link': correlation_matrix_image_link}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/pca/<string:dataset_id>', methods=['POST'])
def apply_pca(dataset_id):
    try:
        # Remove leading and trailing spaces and convert the dataset_id to ObjectId
        object_id = ObjectId(dataset_id.strip())

        # Find the document in the collection
        document = collection.find_one({"_id": object_id})

        if not document:
            return jsonify({'error': 'Dataset no encontrado'}), 404

        # Convert the data to a DataFrame
        df = pd.DataFrame(document['ImputedData'])

        # Check if the dataset has numeric columns for PCA
        numeric_columns = df.select_dtypes(include=['number']).columns
        if len(numeric_columns) == 0:
            return jsonify({'error': 'El dataset no contiene columnas numéricas para aplicar PCA'}), 400

        # Extract numeric data for PCA
        data_for_pca = df[numeric_columns].values

        # Apply PCA
        pca = PCA()
        pca_result = pca.fit_transform(data_for_pca)

        # Create a DataFrame with PCA weights
        pca_weights_df = pd.DataFrame(pca.components_, columns=numeric_columns)

        # Create a new version of the dataset with transformed data
        transformed_data = pd.DataFrame(pca_result, columns=[f'PC{i}' for i in range(1, pca_result.shape[1] + 1)])

        # Add PCAData to the existing dataset in MongoDB
        collection.update_one({"_id": object_id}, {"$set": {"PCAData": transformed_data.to_dict(orient='records')}})

        # Return the PCA weights
        response_data = {
            'pca_weights': pca_weights_df.to_dict(orient='records'),
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)