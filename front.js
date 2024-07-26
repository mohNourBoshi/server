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
        fetch('http://127.0.0.1:5000/image', {
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
                return response.json();
            })
            .then(data => {
                function formatSolve(solve) {
                    console.log(`Input solve: ${solve}`);
                    console.log(solve);
                    if (solve.includes(null)) {
                        console.log("Null values in solve array");
                        return ['1', '+', '1'].join('')
                    }

                    const operators = ['*', '-', '+'];
                    const nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
                    if (operators.includes(solve[1])) {
                        console.log("Middle element is an operator");
                    }
                    else if (operators.includes(solve[2])) {
                        [solve[0], solve[1], solve[2]] = [solve[1], solve[2], solve[0]];
                    } else if (operators.includes(solve[0])) {
                        [solve[0], solve[1], solve[2]] = [solve[2], solve[0], solve[1]];
                    } else if (nums.includes(solve[0]) && nums.includes(solve[1]) && nums.includes(solve[2])) {
                        if (nums.includes(solve[0]) & ['1'].includes(solve[1]) & nums.includes(solve[2])) {
                            solve = solve[0] + '+' + solve[2]

                        }
                        else if (nums.includes(solve[0]) & nums.includes(solve[1]) & ['1'].includes(solve[2])) {
                            solve = solve[1] + '+' + solve[0]

                        }
                        else if (['1'].includes(solve[0]) & nums.includes(solve[1]) & nums.includes(solve[2])) {
                            solve = solve[2] + '+' + solve[1]

                        }
                        else {
                            solve[1] = '+'
                        }
                    }

                    if (['-'].includes(solve[1])) {
                        if (Number(solve[2]) > Number(solve[0])) {
                            solve = solve[0] + '-' + solve[2]
                        }
                    }
                    if (nums.includes(solve[0]) && nums.includes(solve[1]) && nums.includes(solve[2])) {
                        solve = '1+2';
                    }
                    if (solve.includes(null)) {
                        solve = '1+1';
                    }
                    console.log("solve+==" + solve)
                    return solve;
                }
                console.log('Response from server:', data.solvetasks);

                data = data.solvetasks.map(item => item.class_name.trim());
                console.log('Response from server:', data);

                console.log('formate:', formatSolve(data));
                console.log('Response from server:', data);
                console.log('Response from server:', data);
                // Handle response from server if needed
            })
            .catch(error => {
                console.error('Error:', error);
                // Handle error
            });
    };
}


