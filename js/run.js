//audio file
const fileInput = document.getElementById('audio-file');
const fileNameDisplay = document.getElementById('file-name');

fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    fileNameDisplay.textContent = file ? file.name : '';
});
//audio file end

//processing
async function processAudio() {
    const fileInput = document.getElementById('audio-file');
    const fileNameDisplay = document.getElementById('file-name');
    const processButton = document.getElementById('process-btn');
    const resultDisplay = document.getElementById('result');

    fileInput.addEventListener('change', function() {
        fileNameDisplay.textContent = fileInput.files[0].name;
    });

    processButton.addEventListener('click', async function() {
        if (!fileInput.value) {
            alert('Please select a file');
            return;
        }

        processButton.textContent = 'Processing...';
        processButton.disabled = true;

        let model;
        try {
            model = await tf.loadLayersModel('models/model.json'); // Update the path to the model file
        } catch (err) {
            console.error('Failed to load model:', err);
            alert('Failed to load model');
            processButton.textContent = 'PROCESS';
            processButton.disabled = false;
            return;
        }

        const file = fileInput.files[0];
        const audioContext = new AudioContext();
        const audioBuffer = await audioContext.decodeAudioData(await file.arrayBuffer());
        const audioTensor = tf.tidy(() => {
            const waveform = tf.tensor(audioBuffer.getChannelData(0));
            const batched = waveform.reshape([1, -1, 1]);
            const normalized = batched.div(32768.0);
            return normalized;
        });

        let predictionArray;
        try {
            const prediction = await model.predict(audioTensor);
            predictionArray = await prediction.array();
        } catch (err) {
            console.error('Failed to predict:', err);
            alert('Failed to predict');
            processButton.textContent = 'PROCESS';
            processButton.disabled = false;
            return;
        }
        const age = predictionArray[0][0];
        const sex = predictionArray[0][1];
        const weight = predictionArray[0][2];
        const height = predictionArray[0][3];

        let ageLabel = document.getElementById('age-label');
        let sexLabel = document.getElementById('sex-label');
        let weightLabel = document.getElementById('weight-label');
        let heightLabel = document.getElementById('height-label');

        ageLabel.innerHTML = `Age: ${age.toFixed(2)}`;
        sexLabel.innerHTML = `Sex: ${sex.toFixed(2)}`;
        weightLabel.innerHTML = `Weight: ${weight.toFixed(2)}`;
        heightLabel.innerHTML = `Height: ${height.toFixed(2)}`;

        processButton.textContent = 'PROCESS';
        processButton.disabled = false;
    });
}
//end processing