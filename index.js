const fs = require('fs');
const path = require('path');
const https = require('https');
const ffmpeg = require('fluent-ffmpeg');
const tf = require('@tensorflow/tfjs-node');
const posenet = require('@tensorflow-models/posenet');
const { createCanvas, loadImage } = require('canvas');

const VIDEO_URL = 'https://thravos.nyc3.digitaloceanspaces.com/feed/fb87d5b0-00d6-11f0-8796-b313b7f9e6d0-tony%20movie.mov';
const VIDEO_PATH = path.join(__dirname, 'video.mov');
const FRAMES_DIR = path.join(__dirname, 'frames');


function downloadVideo(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https.get(url, (res) => {
      if (res.statusCode !== 200) {
        return reject(new Error(`Download failed with code ${res.statusCode}`));
      }
      res.pipe(file);
      file.on('finish', () => {
        file.close(resolve);
      });
    }).on('error', reject);
  });
}

function extractFrames(videoPath, outputDir, fps = 1) {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir);

    const outputPattern = path.join(outputDir, 'frame-%04d.jpg');

    ffmpeg(videoPath)
      .outputOptions([
        `-vf fps=${fps}`
      ])
      .output(outputPattern)
      .on('start', (cmd) => console.log('Started FFmpeg:', cmd))
      .on('end', () => {
        console.log('Frames extracted.');
        resolve();
      })
      .on('error', (err) => reject(err))
      .run();
  });
}


async function detectPoseOnFrames(dir) {
  const net = await posenet.load();
  const files = fs.readdirSync(dir).filter(f => f.endsWith('.jpg'));
  const results = [];

  for (const file of files) {
    const filePath = path.join(dir, file);
    const img = await loadImage(filePath);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, img.width, img.height);

    const input = tf.browser.fromPixels(canvas);
    const pose = await net.estimateSinglePose(input, {
      flipHorizontal: false, 
      maxDetections: 5,
      scoreThreshold: 0.5
    });
    input.dispose();

    results.push({
      frame: file,
      keypoints: pose.keypoints
    });

    console.log(`Pose extracted from ${file}`);
  }

  const outputPath = path.join(__dirname, 'keypoints.json');
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
  console.log(`Keypoints saved to ${outputPath}`);
}


(async () => {
  try {
    console.log('Downloading video...');
    await downloadVideo(VIDEO_URL, VIDEO_PATH);

    console.log('Extracting frames...');
    await extractFrames(VIDEO_PATH, FRAMES_DIR, 10);

    console.log('Running pose detection...');
    await detectPoseOnFrames(FRAMES_DIR);

    console.log('Done.');
  } catch (err) {
    console.error('Error:', err.message || err);
  }
})();
