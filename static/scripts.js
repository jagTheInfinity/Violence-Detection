document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission

    // Get the file input element
    var fileInput = document.getElementById('videoFile');
    
    // Check if a file is selected
    if (fileInput.files.length > 0) {
        // Create a FormData object to store the file data
        var formData = new FormData();
        formData.append('file', fileInput.files[0]); // Append the selected file to the FormData object

        // Send a POST request to the server with the file data
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Handle the response from the server (e.g., display results)
            console.log(data);  // Add this line to check the data received
            var resultDiv = document.getElementById('result');
            resultDiv.innerHTML = '';
            var message = data.fight ? 'Fight detected!' : 'No fight detected.';
            var percentage = parseFloat(data.precentegeoffight) * 100;
            var processingTime = data.processing_time;
            resultDiv.innerHTML = '<p>' + message + '</p>';
            resultDiv.innerHTML += '<p>Confidence: ' + percentage.toFixed(2) + '%</p>';
            resultDiv.innerHTML += '<p>Processing Time: ' + processingTime + ' milliseconds</p>'; // Log the response data to the console for debugging
        })
        .catch(error => {
            console.error('Error:', error); // Log any errors to the console
        });
    } else {
        // If no file is selected, display an error message to the user
        alert('Please select a file before uploading.');
    }
});

function plotConfidence(percentage) {
    // Prepare data for plotting
    var confidenceData = parseFloat(percentage) * 100;

    // Create the plot
    var ctx = document.getElementById('confidencePlot').getContext('2d');
    var chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Confidence'],
            datasets: [{
                label: 'Percentage',
                data: [confidenceData],
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}