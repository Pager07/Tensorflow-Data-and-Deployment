const  IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const TRAIN_TEST_RATIO = 5/6;

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
                canvas.height = img.chuckSize
                
                //Split: Process 5000 rows each time
                for(let i = 0; i< NUM_DATASET_ELEMENTS/ chuckSize; i++){
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
                    const datasetBytesView = new Float32Array(
                        datasetBytesBuffer,
                        i*IMAGE_SIZE*chunkSize*4,
                        IMAGE_SIZE * chunkSize);

                    ctx.drawImage(img,0,i*chunkSize,img.width, chunkSize,0,0,img,width,
                        chuckSize);

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
                        datasetBytesView[j] = imageData.data[j * 4] / 255;
                    }

                }
            }
            img.src = MINST_IMAGES_SPRITE_PATH
        });
    }
    nextTrainBatch(){}
    nextTestBatch(){}
}