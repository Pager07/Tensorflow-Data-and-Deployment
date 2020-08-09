import {MnistData} from './data.js';
var canvas,ctx,saveButton,clearButton;
var pos = {x:0,y:0};
var rawImage;
var model;

function getModel(){
    //Very straight forward. 
    //1 question is: how does this script have access to tf variable?
    model = tf.sequential();
    model.add(tf.layers.conv2d({filters:8 ,kernelSize:3, activation:'relu',
         inputShape:[28,28,1]}));
    model.add(tf.layers.maxPooling2d({poolSize:[2,2]}));
    model.add(tf.layers.conv2d({filters:16,kernelSize:3, activation:'relu'}));
    model.add(tf.layers.maxPooling2d({poolSize:[2,2]}));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({units:128,activation:'relu'}));
    model.add(tf.layers.dense({units:10, activation:'softmax'}));

    model.compile({loss:'categoricalCrossentropy', optimizer:tf.train.adam(),
            metrics:['accuracy']});
    
    return model
}

async function train(model,data){
    const metrics = ['loss', 'val_loss' , 'accuracy', 'val_accuracy'];
    const container = {name:'Model Training' , styles: {height: '640px'}};
    const fitCallbacks = tfvis.show.fitCallbacks(container , metrics);

    const BATCH_SIZE =512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE  = 1000;
    
    const [trainXs, trainYs] = tf.tidy(()=>{
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE,28,28,1]),
            d.labels
        ];
    });
    
    const [testXs,testYs] = tf.tidy(()=>{
        const d = data.nextTrainBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE,28,28,1]),
            d.labels
        ]
    });

    return model.fit(trainXs, trainYs,{
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 20,
        shuffle: true,
        callbacks: fitCallbacks

    });

}

function setPosition(e){
    //pos is a dictionay holding cooridnates
    pos.x = e.clientX-100;
    pos.y = e.clientY-100;
}

function draw(e){
    if(e.buttons!=1) return;
    //The beginPath() clears the current path drawing state. 
    //The beginPath() function can be used to begin a new path, 
    ctx.beginPath();
    //Styling:
    //Draw a line with rounded end caps
    //strokeStyle property sets or returns the color
    ctx.lineWidth = 24;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';
    //I dont understand the below code
    ctx.moveTo(pos.x,pos.y);
    setPosition(e);
    ctx.lineTo(pos.x,pos.y);
    ctx.stroke();
    //What is rawImage?
    // it is a reference to the image object inside the canvas
    rawImage.src = canvas.toDataURL('img/png');

}

function erase(){
    ctx.fillStyle = 'black';
    ctx.fillRect(0,0,280,280);
}

function save(){
    //get the image in tensor form from canvas, 1 is the number of channle to get
    //More at:https://js.tensorflow.org/api/1.2.8/#browser.fromPixels
    var raw = tf.browser.fromPixels(rawImage,1);
    var resized = tf.image.resizeBilinear(raw,[28,28]);
    var tensor = resized.expandDims(0);
    var prediction = model.predict(tensor);
    //Datasync means, get/download the values from in the tensor
    //More at: https://js.tensorflow.org/api/1.2.8/#tf.Tensor.dataSync
    var pIndex = tf.argMax(prediction,1).dataSync();
    alert(pIndex);
}

function init(){
    //Get the canvas and the image object
    canvas = document.getElementById('canvas');
    rawImage = document.getElementById('canvasimg');

    //Initialize the canvas to black
    ctx = canvas.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0,0,280,280);

    //Start drawing when mouse is moving and being clicked
    // We are checking for mouse hold inside the function
    canvas.addEventListener('mousemove',draw);

    // Update/set the position of the mouse when ever it:
    // mouse down or mouse enters the canvas
    canvas.addEventListener('mousedown', setPosition);
    canvas.addEventListener('mouseenter',setPosition);

    //Save button
    saveButton = document.getElementById('sb');
    saveButton.addEventListener('click',save);

    //Clear button
    clearButton = document.getElementById('cb');
    clearButton.addEventListener('click',erase);
}
async function run(){
    console.log('DOMContentLoaded!!');
    const data = new MnistData();
    
    console.log('Starting to load data....');
    await data.load();
    console.log('Data has be loaded!');
    
    console.log('Initializing the model....');
    const model = getModel();
    console.log('Model has been initalized!');
    
    tfvis.show.modelSummary({name:'Model Architecture'}, model);
    
    console.log('Training Model.......');
    await train(model,data);
    console.log('Model has finished training!!');
    
    init();
    alert('Trainingg is done, Try classifying your handwriting!');
}

document.addEventListener('DOMContentLoaded',run);