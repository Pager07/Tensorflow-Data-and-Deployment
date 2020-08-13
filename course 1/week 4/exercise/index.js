let mobilenet;
let model;
//pass in a video element 
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var rockSamples=0, paperSamples=0, scissorsSamples = 0;
let isPredicting =  false;

async function loadMobilenet(){
    const mobilenet = 
        await tf.loadLayersModel(
         'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    // tf.model allows us to apply a view to any other given model
    return tf.model({inputs:mobilenet.inputs, outputs:layer.output});
}

async function train(){
    //reset the one-hot encoded arrray
    dataset.ys = null;
    //set the this.ys using the this.labels 
    //one of my mistake was that: I passed activations instead of activation 
    // so my output values where not in 0-1 range
    dataset.encodeLabels(3);
    model = tf.sequential({
        layers: [
            tf.layers.flatten({inputShape:mobilenet.outputs[0].shape.slice(1)}),
            tf.layers.dense({units:100, activation:'relu'}),
            tf.layers.dense({units:3, activation:'softmax'})
        ]
    });
    const optimizer = tf.train.adam(0.0001);
    model.compile({optimizer: optimizer , loss: 'categoricalCrossentropy'});
    let loss = 0
    model.fit(dataset.xs, dataset.ys, {
        epochs: 10,
        callbacks: {
            onBatchEnd: async (batch, logs)=>{
                //.toFixed is a js function that rounds loss to 5 dp
                loss = logs.loss.toFixed(5);
                console.log('LOSS:' + loss);
            }
        }
    });

}

function handleButton(elem){
    switch(elem.id){
        case "0": 
            rockSamples = rockSamples + 1
            document.getElementById('rocksamples').innerText = "Rock Samples: " + rockSamples;
            break 
        case "1":
            paperSamples++;
            document.getElementById('papersamples').innerText = 'Paper Samples: ' + paperSamples;
            break
        case "2":
            scissorsSamples++;
            document.getElementById('scissorssamples').innerText = 'Scissors Samples: ' + scissorsSamples;
            break
    }
    label = parseInt(elem.id);
    const img = webcam.capture();
    //mobilenet.predict(img) returns embbeding of a certain layer?
    // Yes, mobilenet is not the raw model from cdn, we made a new model
    // where it's the chopped up version of mobilenet
    dataset.addExample(mobilenet.predict(img), label);
}

async function predict(){
    while(isPredicting){
        const predictedClass = tf.tidy(()=>{
            const img = webcam.capture();
            const activations = mobilenet.predict(img);
            const predictions = model.predict(activations);
            return predictions.as1D().argMax();

        });
        const classId = (await predictedClass.data())[0];
        
        var predictionText = "";
        switch(classId){
            case 0:
                predictionText = "I see rock";
                break;
            case 1:
                predictionText = "I see paper";
                break;
            case 2:
                predictionText = "I see scissors";
                break;
        }
        document.getElementById('prediction').innerText = predictionText;
        predictedClass.dispose()
        //Calling tf.nextFrame() in a browser context will release the UI thread 
        //so the page can be responsive.
        await tf.nextFrame();
    }

}

function doTraining(){
    train();
}

function startPredicting(){
    isPredicting = true;
    predict();
}

function stopPredicting(){
    isPredicting = false;
    predict();
}

async function init(){
    await webcam.setup();
    mobilenet = await loadMobilenet();
    tf.tidy(()=> mobilenet.predict(webcam.capture()));
}

init();