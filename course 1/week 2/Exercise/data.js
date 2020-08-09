const  IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const TRAIN_TEST_RATIO = 5/6;
const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MINST_IMAGES_SPRITE_PATH = 
'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';

const MINST_LABELS_PATH = 
'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';
/**
 * A class that fetechs the sprited MNIST dataset
 * RETURNS SHUFFLED BATCHES
 * NOTE: This will get much easier. For now, we do data fetching and manipulation manually.
 */
export class MnistData{
    constructor(){
        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;
    }
    async load(){
        /**
         * This function will do the following
         * 1. download the sprite and slice it
         * 2. download the lables and decode them
         */
        
        /**
         * Make a request for the MNIST sprited image.
         * The data has to be loaded to the Image() object, then the object is placed in
         *  the canvas
         */ 
        
        const img = new Image();
        const canvas = document.createElement('canvas');
        //Tell js that canvas is going to be 2d
        const ctx = canvas.getContext('2d');
        const imgRequest = new Promise((resolve,reject)=>{
            img.crossOrigin = '';
            //slice the image when loaded
            img.onload = ()=>{
                // set the object height and width equal to image height and width
                img.width = img.naturalWidth;
                img.height = img.naturalHeight;
                
                //some buffer initialization: MULT BY 4 because js will loaded it as 
                //rbga image even though it's a gray image. By setting all the channels to 
                //equal
                // Create ArrayBuffer aka. get a chunk of memory location.It stores binary data.
                // We can combine single buffer with multiple views of different types, 
                //Size of the buffer: (Area_of_rectangle * 4) bytes
                //Imagine this as a rectangle 
                // More at : 
                //https://stackoverflow.com/questions/11554006/javascript-arraybuffer-whats-it-for
                //https://developer.mozilla.org/en-US/docs/Web/JavaScript/Typed_arrays#Working_with_complex_data_structures
                const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS*IMAGE_SIZE*4)
                //Setting the canvas size
                const  chunkSize = 5000;
                canvas.width = img.width;
                canvas.height = chunkSize;
                
                //Split: Process 5000 rows each time
                for(let i = 0; i< NUM_DATASET_ELEMENTS/ chunkSize; i++){
                    //Create a list to store pixel values. We cant use normal list
                    //glMatrix was developed primarily for WebGL, which requires 
                    //that vectors and matrices be passed as Float32Array. We are using WebGl for GPU access
                    // Creating a float32Array view for a buffer: 
                    //Flot32Array(buffer,offset,lengthOfBuffer)
                    // What does offset do?: imagine a smaller block (Image_Size * chunkSize)
                    // 1*(smallerblock), 2*(smaller_block)....
                    // it lets the choose the block/part to view in the buffer
                    // What does lengthOfBuffer: this refers to length of the array
                    // We are going to store only one channel, so the size is only (IMAGE_SIZE*chunkSize)
                    //More at: https://webplatform.github.io/docs/javascript/Float32Array/
                    const datasetBytesView = new Float32Array(
                        datasetBytesBuffer,
                        i*IMAGE_SIZE*chunkSize*4,
                        IMAGE_SIZE * chunkSize);
                    
                    //Put part of the image to the canvas
                    //More at: 
                    //https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/drawImage
                    ctx.drawImage(img,0,i*chunkSize,img.width,chunkSize,
                        0,0,img.width,chunkSize);

                    //Get An ImageData object containing the image data for the rectangle 
                    //of the canvas specified.
                    // The imageData.data is a flatten array, holding pixels values of all
                    // the channels. eg.[R1,G1,B1,A1.....Rn,Gn,Bn,An].
                    //More at:  https://www.w3schools.com/tags/canvas_getimagedata.asp
                    const imageData = ctx.getImageData(0,0,canvas.width,canvas.height);
                    
                    // We are looping over each pixel 
                    // imageData.data.length /4 is the number of pixels in the the list
                    for(let j =0; j< imageData.data.length / 4 ; j++){
                        //All channels hold an equal value since the image is grayscale,
                        //so just read  the red channel.
                        // The imageData.data is a flatten array, holding pixels values of all
                        // the channels. eg.[R1,G1,B1,A1.....Rn,Gn,Bn,An].
                        // As we only need 1 channel per pixel j*4
                        datasetBytesView[j] = imageData.data[j * 4] / 255;
                    }

                }
                //Make a new view that view all the buffer holding images data
                this.datasetImages = new Float32Array(datasetBytesBuffer);
                resolve();

            };
            img.src = MINST_IMAGES_SPRITE_PATH
        });

        //Use javascript fetch to get object of promise that will return labels
        const labelsRequest = fetch(MINST_LABELS_PATH);

        //imgResponse and labelsReponse are objects of promise types.
        // Be aware of 2 things:
        // First, we have to use "await" to wait for them to finish
        // Second, return stuffs. by either Resolving or Rejecting 
        const [imgResponse, labelsReponse] = 
            await Promise.all([imgRequest, labelsRequest])
        
       this.datasetLabels = new Uint8Array(await labelsReponse.arrayBuffer());

        //Create shuffled indices into the train/test set for when we select a
        // random dataset element for training / validation
        this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
        this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

        //Slice the images and labels into ttrain and teset sets.
        //Slice it from the view/array
        this.trainImages  =
            this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
        
        this.testImages = 
            this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
        
        this.trainLabels = 
            this.datasetLabels.slice(0, IMAGE_SIZE*NUM_TRAIN_ELEMENTS);
        
        this.testLabels = 
            this.datasetLabels.slice(IMAGE_SIZE*NUM_TRAIN_ELEMENTS);
        

    }
    nextTrainBatch(batchSize){
        // Passes the array holding training images and labels
        return this.nextBatch(
            batchSize, [this.trainImages, this.trainLabels],()=>{
                this.shuffledTrainIndex = 
                    (this.shuffledTrainIndex+1) % this.trainIndices.length;
                return this.trainIndices[this.shuffledTrainIndex]
            }
        );
    }
    nextTestBatch(batchSize){
        return this.nextBatch(
            batchSize, [this.testImages, this.testLabels], ()=>{
                // find the index of the needed item ??
                // I fail to understant moduls maths that happening here
                this.shuffledTestIndex =
                    (this.shuffledTestIndex +1) % this.testIndices.length;
                
                // get the item/index
                return this.testIndices[this.shuffledTestIndex];
            }
        );
    }
    nextBatch(batchSize, data, index){
        const batchImagesArray = new Float32Array(batchSize*IMAGE_SIZE);
        const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

        for(let i =0; i<batchSize; i++){
            //get the row number in array AKA X|Y
            const idx = index();
            //Why does slicing index start from idx*Image_size?
            // It's whole alot easier, to see things as matrix.
            //idx*IMAGE_SIZE represents the start of the row, while idx*IMAGE_SIZE+IMAGE_SIZE
            // represents end of the row.
            //Why does slicing index end at id*Image_size+Image_size?
            //What does data[0] look like?: It can be array holding images for train|test
            const image =
                data[0].slice(idx*IMAGE_SIZE, idx*IMAGE_SIZE+IMAGE_SIZE);
            //i*Image_Size points to the ith row
            batchImagesArray.set(image, i*IMAGE_SIZE);

            const label = 
                data[1].slice(idx*NUM_CLASSES , idx*NUM_CLASSES+NUM_CLASSES);
            batchLabelsArray.set(label, i*NUM_CLASSES);
        }
        // The batchImagesArray is of shape [batchSize , Image_Size]        
        const xs = tf.tensor2d(batchImagesArray,[batchSize,IMAGE_SIZE]);
        const labels = tf.tensor2d(batchLabelsArray,[batchSize, NUM_CLASSES]);

        return {xs, labels}

    }
}