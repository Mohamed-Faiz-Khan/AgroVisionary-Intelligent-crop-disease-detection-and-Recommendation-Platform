document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('cropForm');
    const submitBtn = document.getElementById('submitBtn');

    const ranges = {
        "Nitrogen": [0, 200],
        "Phosporus": [5, 145],
        "Potassium": [5, 205],
        "Temperature": [10, 44],
        "Humidity": [14, 85],
        "Ph": [3, 9],
        "Rainfall": [20, 300]
    };

    const inputs = form.querySelectorAll('input[type="number"]');

    inputs.forEach(input => {
        input.addEventListener('input', function () {
            validateInput(input);
            checkAllInputs();
        });
    });

    function validateInput(input) {
        const value = parseFloat(input.value);
        const range = ranges[input.id];
        const errorMsg = document.getElementById(input.id + 'Error');

        if (value < range[0] || value > range[1]) {
            errorMsg.textContent = `Please enter within the specific range (${range[0]} - ${range[1]})`;
            errorMsg.style.display = 'block';
            input.classList.add('is-invalid');
        } else {
            errorMsg.style.display = 'none';
            input.classList.remove('is-invalid');
        }
    }

    function checkAllInputs() {
        const invalidInputs = form.querySelectorAll('.is-invalid');
        submitBtn.disabled = invalidInputs.length > 0;
    }
});
