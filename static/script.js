// Get the video and canvas elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Load the coco-ssd model
const modelPromise = cocoSsd.load();

navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then((stream) => {
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            updatePredictions();
        };
    })
    .catch((error) => {
        console.error('Error accessing the camera: ', error);
    });

video.addEventListener('loadeddata', updatePredictions);


// Function to update the predictions
async function updatePredictions() {
    const model = await modelPromise;
    const predictions = await model.detect(video);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    for (let i = 0; i < predictions.length; i++) {
        const prediction = predictions[i];

        if (prediction.class === 'cell phone') {
            ctx.fillStyle = 'white'; 
            ctx.fillRect(prediction.bbox[0], prediction.bbox[1], prediction.bbox[2], prediction.bbox[3]);
        } else {
            ctx.beginPath();
            ctx.rect(prediction.bbox[0], prediction.bbox[1], prediction.bbox[2], prediction.bbox[3]);
            ctx.lineWidth = 2;
            ctx.strokeStyle = 'red';
            ctx.fillStyle = 'red';
            ctx.stroke();
    
            ctx.fillText(
                `${prediction.class} (${Math.round(prediction.score * 100)}%)`,
                prediction.bbox[0],
                prediction.bbox[1] > 10 ? prediction.bbox[1] - 5 : 10
            );
        }
    }

    requestAnimationFrame(updatePredictions);
}

updatePredictions();