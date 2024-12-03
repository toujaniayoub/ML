// static/js/script.js

document.getElementById('prediction-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Collect form data
    let formData = new FormData(this);
    let data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });
    
    // Send the data to the backend for prediction
    fetch('/predict', {
        method: 'POST',
        body: new URLSearchParams(data),
    })
    .then(response => response.json())
    .then(data => {
        // Show the prediction result
        document.getElementById('predicted-os').textContent = data.prediction;
        document.getElementById('result').style.display = 'block';
    })
    .catch(error => console.error('Error:', error));
});
