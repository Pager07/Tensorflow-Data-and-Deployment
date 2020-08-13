class RPSDataset{
    //RPS stands for rock paper siccors 
    constructor(){
        // this.labels will hold the raw labels 
        this.labels = []
        // we will also use another class variable this.ys to store the one-encoded version
    }

    addExample(example,label){
        if(this.xs == null){
            this.xs = tf.keep(example)
            this.labels.push(label);
        }else{
            const oldX = this.xs
            this.xs = tf.keep(oldX.concat(example,0))
            this.labels.push(label);
            oldX.dispose();
        }
    }

    encodeLabels(numClasses){
        //This function will produce this.ys from this.labels
        //Keep a list of labels, and only create the much larger list of 
        //one-hot encoded ones before you train.
        //Creating one-hot encoded vectors is very ineffiecnt
        //What does tf.keep do?
        //Keeps a tf.Tensor generated inside a tf.tidy() from being disposed automatically.
        //More at:https://js.tensorflow.org/api/0.11.2/#keep
        for(var i = 0 ; i< this.labels.length; i++){
            if(this.ys == null){
                this.ys = tf.keep(tf.tidy(
                    ()=>{return tf.oneHot(tf.tensor1d([this.labels[i]]).toInt(),numClasses)}
                ));
            }else{
                const y = tf.tidy(()=>{
                    return tf.oneHot(tf.tensor1d([this.labels[i]]).toInt(), numClasses);
                });
                const oldY = this.ys;
                //Try removing the keep and see what happens 
                this.ys = tf.keep(oldY.concat(y, 0));
                oldY.dispose();
                y.dispose();
            }
        }
        
    }
}
  
