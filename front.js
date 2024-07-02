let htmltxt = `
<input type="file" id="imageInput" accept="image/*">
<button onclick="uploadImage()">Upload Image</button>
`
document.querySelector('body').innerHTML = htmltxt

function uploadImage() {
    const input = document.getElementById('imageInput');
    const file = input.files[0];

    // Check if a file is selected
    if (!file) {
        alert('Please select an image.');
        return;
    }

    // Read the selected image file
    const reader = new FileReader();
    reader.readAsDataURL(file);

    reader.onload = function (event) {
        const base64Image = event.target.result;

        // Construct the payload
        const payload = {
            "base64_image": base64Image
        };

        // Send the payload as JSON via fetch
        // fetch('https://nour.mooo.info/image', {
        // fetch('http://38.242.235.57:5050/image', {
        // fetch('https://5000-cs-941090012827-default.cs-europe-west4-fycr.cloudshell.dev/image', {
        fetch('http://127.0.0.1:5000/imagr', {
            // fetch('https://chaptcha.onrender.com/image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.text();
            })
            .then(data => {
                console.log('Response from server:', data);
                // Handle response from server if needed
            })
            .catch(error => {
                console.error('Error:', error);
                // Handle error
            });
    };
}


