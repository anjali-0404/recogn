# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


from flask import Flask, request, jsonify
import inference_classifier  # Import necessary modules

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('input_data')  # Receive input from Flutter
    result = inference_classifier.some_function(data)  # Call function from another file
    return jsonify({"result": result})  # Return result to Flutter

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
