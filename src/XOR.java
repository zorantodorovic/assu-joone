

/**
 * Created by zorantodorovic on 09/01/2017.
 */


import org.joone.engine.*;
import org.joone.engine.learning.TeachingSynapse;
import org.joone.io.FileInputSynapse;
import org.joone.io.FileOutputSynapse;
import org.joone.io.MemoryOutputSynapse;
import org.joone.net.NeuralNet;


public class XOR implements NeuralNetListener {
    private NeuralNet nnet1 = null;
    private NeuralNet nnet2 = null;
    private FileInputSynapse firstInputSynapse;
    private FileInputSynapse secondInputSynapse;
    private FileInputSynapse desiredFirstOutputSynapse;
    private FileInputSynapse desiredSecondOutputSynapse;
    private MemoryOutputSynapse firstOutputSynapse;
    private MemoryOutputSynapse secondOutputSynapse;

    public XOR() {
    }

    public static void main(String[] args) {
        XOR xor = new XOR();
        xor.initFirstNN();
        System.out.println("First NN initialized");
        xor.trainFirstNN();
        System.out.println("First NN trained");
        xor.interrogateFirstNN();
        System.out.println("First NN interrogated");

        xor.initSecondNN();
        System.out.println("Second NN initialized");
        xor.trainSecondNN();
        System.out.println("Second NN trained");
        xor.interrogateSecondNN();
        System.out.println("Second NN interrogated");

        xor.testFirstNN();
    }


    public void trainFirstNN() {
        this.firstInputSynapse.setFileName("in_first.txt");
        this.firstInputSynapse.setAdvancedColumnSelector("1,2,3");
        this.desiredFirstOutputSynapse.setFileName("in_first.txt");
        this.desiredFirstOutputSynapse.setAdvancedColumnSelector("4,5");
        Monitor monitor = this.nnet1.getMonitor();
        monitor.setLearningRate(0.8D);
        monitor.setMomentum(0.3D);
        monitor.setTrainingPatterns(3);
        monitor.setTotCicles(10000);
        monitor.setLearning(true);
        this.nnet1.addNeuralNetListener(this);
        this.nnet1.start();
        this.nnet1.getMonitor().Go();
        this.nnet1.join();
        System.out.println("Network stopped. Last RMSE=" + this.nnet1.getMonitor().getGlobalError());
//        this.interrogateFirstNN();
    }

    public void trainSecondNN() {
//        this.inputSynapse.setInputArray(this.inputArray);
        this.secondInputSynapse.setFileName("in_first.txt");
        this.secondInputSynapse.setAdvancedColumnSelector("4,5");
//        this.desiredOutputSynapse.setInputArray(this.desiredOutputArray);
        this.desiredSecondOutputSynapse.setFileName("in_first.txt");
        this.desiredSecondOutputSynapse.setAdvancedColumnSelector("1,2,3");
        Monitor monitor = this.nnet2.getMonitor();
        monitor.setLearningRate(0.8D);
        monitor.setMomentum(0.3D);
        monitor.setTrainingPatterns(3);
        monitor.setTotCicles(10000);
        monitor.setLearning(true);
        this.nnet2.addNeuralNetListener(this);
        this.nnet2.start();
        this.nnet2.getMonitor().Go();
        this.nnet2.join();
        System.out.println("Network stopped. Last RMSE=" + this.nnet2.getMonitor().getGlobalError());
        System.out.println("Second network interrogation started");
//        this.interrogateSecondNN();
    }

    private void interrogateFirstNN() {
        Monitor monitor = this.nnet1.getMonitor();
        monitor.setTrainingPatterns(3);
        monitor.setTotCicles(1);
        monitor.setLearning(false);


        FileOutputSynapse output = new FileOutputSynapse();
        output.setFileName("test_out_first.txt");
        if(this.nnet1 != null) {
            this.nnet1.addOutputSynapse(output);
            System.out.println(this.nnet1.check());
            this.nnet1.start();
            monitor.Go();
            this.nnet1.join();
        }
    }

    private void interrogateSecondNN() {
        Monitor monitor = this.nnet2.getMonitor();
        monitor.setTrainingPatterns(3);
        monitor.setTotCicles(1);
        monitor.setLearning(false);
        FileOutputSynapse output = new FileOutputSynapse();
        output.setFileName("out_second.txt");
        if(this.nnet2 != null) {
            this.nnet2.addOutputSynapse(output);
            System.out.println(this.nnet2.check());
            this.nnet2.start();
            monitor.Go();
            this.nnet2.join();
        }
    }

    private void testFirstNN() {
        Layer input = this.nnet1.getInputLayer();
        input.removeAllInputs();
        input.setLayerName("L.input");
        input.setRows(3);

        FileInputSynapse fileInputSynapse = new FileInputSynapse();
        fileInputSynapse.setFileName("test_in_first.txt");
        fileInputSynapse.setAdvancedColumnSelector("1,2,3");

        Monitor monitor = this.nnet1.getMonitor();
        monitor.setTrainingPatterns(3);
        monitor.setTotCicles(1);
        monitor.setLearning(false);


        FileOutputSynapse fileOutputSynapse = new FileOutputSynapse();
        fileOutputSynapse.setFileName("test_out_first.txt");
        if(this.nnet1 != null) {
            this.nnet1.addInputSynapse(fileInputSynapse);
            this.nnet1.addOutputSynapse(fileOutputSynapse);
            System.out.println(this.nnet1.check());
            this.nnet1.start();
            monitor.Go();
            this.nnet1.join();
        }

    }

    protected void initFirstNN() {
        LinearLayer input = new LinearLayer();
        SigmoidLayer hidden = new SigmoidLayer();
        SigmoidLayer output = new SigmoidLayer();
        input.setRows(3);
        hidden.setRows(5);
        output.setRows(2);
        input.setLayerName("L.input");
        hidden.setLayerName("L.hidden");
        output.setLayerName("L.output");

        FullSynapse synapse_IH = new FullSynapse();
        FullSynapse synapse_HO = new FullSynapse();
        input.addOutputSynapse(synapse_IH);
        hidden.addInputSynapse(synapse_IH);
        hidden.addOutputSynapse(synapse_HO);
        output.addInputSynapse(synapse_HO);

        this.firstInputSynapse = new FileInputSynapse();
        input.addInputSynapse(this.firstInputSynapse);
        this.firstOutputSynapse = new MemoryOutputSynapse();
        output.addOutputSynapse(this.firstOutputSynapse);
//        this.desiredOutputSynapse = new MemoryInputSynapse();
        TeachingSynapse trainer = new TeachingSynapse();
        this.desiredFirstOutputSynapse = new FileInputSynapse();
        trainer.setDesired(this.desiredFirstOutputSynapse);
        this.nnet1 = new NeuralNet();
        this.nnet1.addLayer(input, 0);
        this.nnet1.addLayer(hidden, 1);
        this.nnet1.addLayer(output, 2);
        this.nnet1.setTeacher(trainer);
        output.addOutputSynapse(trainer);
    }

    protected void initSecondNN(){
        LinearLayer input = new LinearLayer();
        SigmoidLayer hidden = new SigmoidLayer();
        SigmoidLayer output = new SigmoidLayer();
        input.setRows(2);
        hidden.setRows(5);
        output.setRows(3);
        input.setLayerName("L2.input");
        hidden.setLayerName("L2.hidden");
        output.setLayerName("L2.output");

        FullSynapse synapse_IH = new FullSynapse();
        FullSynapse synapse_HO = new FullSynapse();
        input.addOutputSynapse(synapse_IH);
        hidden.addInputSynapse(synapse_IH);
        hidden.addOutputSynapse(synapse_HO);
        output.addInputSynapse(synapse_HO);

        this.secondInputSynapse = new FileInputSynapse();
        input.addInputSynapse(this.secondInputSynapse);
        this.secondOutputSynapse = new MemoryOutputSynapse();
        output.addOutputSynapse(this.secondOutputSynapse);
//        this.desiredOutputSynapse = new MemoryInputSynapse();
        TeachingSynapse trainer = new TeachingSynapse();
        this.desiredSecondOutputSynapse = new FileInputSynapse();
        trainer.setDesired(this.desiredSecondOutputSynapse);
        this.nnet2 = new NeuralNet();
        this.nnet2.addLayer(input, 0);
        this.nnet2.addLayer(hidden, 1);
        this.nnet2.addLayer(output, 2);
        this.nnet2.setTeacher(trainer);
        output.addOutputSynapse(trainer);
    }


    public void cicleTerminated(NeuralNetEvent e) {
    }

    public void errorChanged(NeuralNetEvent e) {
        Monitor mon = (Monitor)e.getSource();
        System.out.println("Cycle: " + (mon.getTotCicles() - mon.getCurrentCicle()) + " RMSE:" + mon.getGlobalError());
    }

    public void netStarted(NeuralNetEvent e) {
    }

    public void netStopped(NeuralNetEvent e) {
        System.out.println("Stopped.");
    }

    public void netStoppedError(NeuralNetEvent e, String error) {
    }
}

