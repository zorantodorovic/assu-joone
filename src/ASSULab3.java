/**
 * Created by zorantodorovic on 09/01/2017.
 */


import org.joone.engine.*;
import org.joone.engine.learning.TeachingSynapse;
import org.joone.io.FileInputSynapse;
import org.joone.io.FileOutputSynapse;
import org.joone.io.MemoryOutputSynapse;
import org.joone.net.NeuralNet;

public class ASSULab3 implements NeuralNetListener {
    private NeuralNet nnet1 = null;
    private NeuralNet nnet2 = null;
    private FileInputSynapse firstInputSynapse;
    private FileInputSynapse secondInputSynapse;
    private FileInputSynapse desiredFirstOutputSynapse;
    private FileInputSynapse desiredSecondOutputSynapse;
    private MemoryOutputSynapse firstOutputSynapse;
    private MemoryOutputSynapse secondOutputSynapse;

    private Integer numberOfCycles = 10000;
    private Integer numberOfTrainingPatterns = 3;
    private Double learningRate = 0.8;
    private Double momentum = 0.3;
    private Integer numberOfTestCycles = 10;

    private ASSULab3() {}

    public static void main(String[] args) {
        ASSULab3 assu = new ASSULab3();
        assu.initFirstNN();
        System.out.println("First NN initialized");
        assu.trainFirstNN();
        System.out.println("First NN trained");
        assu.interrogateFirstNN();
        System.out.println("First NN interrogated");

        assu.initSecondNN();
        System.out.println("Second NN initialized");
        assu.trainSecondNN();
        System.out.println("Second NN trained");
        assu.interrogateSecondNN();
        System.out.println("Second NN interrogated");

        assu.testFirstNN();
    }


    private void trainFirstNN() {
        this.firstInputSynapse.setFileName("inputData.txt");
        this.firstInputSynapse.setAdvancedColumnSelector("1,2,3");
        this.desiredFirstOutputSynapse.setFileName("inputData.txt");
        this.desiredFirstOutputSynapse.setAdvancedColumnSelector("4,5");
        Monitor monitor = this.nnet1.getMonitor();
        monitor.setLearningRate(learningRate);
        monitor.setMomentum(momentum);
        monitor.setTrainingPatterns(numberOfTrainingPatterns);
        monitor.setTotCicles(numberOfCycles);
        monitor.setLearning(true);
        this.nnet1.addNeuralNetListener(this);
        this.nnet1.start();
        this.nnet1.getMonitor().Go();
        this.nnet1.join();
        System.out.println("Network stopped. Last RMSE=" + this.nnet1.getMonitor().getGlobalError());
//        this.interrogateFirstNN();
    }

    private void trainSecondNN() {
        this.secondInputSynapse.setFileName("inputData.txt");
        this.secondInputSynapse.setAdvancedColumnSelector("4,5");
        this.desiredSecondOutputSynapse.setFileName("inputData.txt");
        this.desiredSecondOutputSynapse.setAdvancedColumnSelector("1,2,3");
        Monitor monitor = this.nnet2.getMonitor();
        monitor.setLearningRate(learningRate);
        monitor.setMomentum(momentum);
        monitor.setTrainingPatterns(numberOfTrainingPatterns);
        monitor.setTotCicles(numberOfCycles);
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
        monitor.setTrainingPatterns(numberOfTrainingPatterns);
        monitor.setTotCicles(1);
        monitor.setLearning(false);


        FileOutputSynapse output = new FileOutputSynapse();
        output.setFileName("interrogation_out_firstNN.txt");
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
        monitor.setTrainingPatterns(numberOfTrainingPatterns);
        monitor.setTotCicles(1);
        monitor.setLearning(false);
        FileOutputSynapse output = new FileOutputSynapse();
        output.setFileName("interrogation_out_secondNN.txt");
        if(this.nnet2 != null) {
            this.nnet2.addOutputSynapse(output);
            System.out.println(this.nnet2.check());
            this.nnet2.start();
            monitor.Go();
            this.nnet2.join();
        }
    }

    private void testFirstNN() {
        String fileName;
        for (int i = 0; i < numberOfTestCycles; i++) {
            if (i == 0) {
                fileName = "test_in_first.txt";
            } else {
                fileName = "test_out_middle.txt";
            }
            Layer input = this.nnet1.getInputLayer();
            input.removeAllInputs();
            input.setLayerName("L.input");
            input.setRows(3);

            FileInputSynapse fileInputSynapse = new FileInputSynapse();
            fileInputSynapse.setFileName(fileName);
            fileInputSynapse.setAdvancedColumnSelector("1,2,3");

            Monitor monitor = this.nnet1.getMonitor();
            monitor.setTrainingPatterns(numberOfTrainingPatterns);
            monitor.setTotCicles(1);
            monitor.setLearning(false);

            FileOutputSynapse fileOutputSynapse = new FileOutputSynapse();
            fileOutputSynapse.setFileName("test_out_first.txt");
            if (this.nnet1 != null) {
                this.nnet1.addInputSynapse(fileInputSynapse);
                this.nnet1.addOutputSynapse(fileOutputSynapse);
                System.out.println(this.nnet1.check());
                this.nnet1.start();
                monitor.Go();
                this.nnet1.join();
            }
            testSecondNN();
        }
    }

    private void testSecondNN() {
        Layer input2 = this.nnet2.getInputLayer();
        input2.removeAllInputs();
        input2.setLayerName("L2.input");
        input2.setRows(2);
        FileInputSynapse fileInputSynapse2 = new FileInputSynapse();
        fileInputSynapse2.setFileName("test_out_first.txt");
        fileInputSynapse2.setAdvancedColumnSelector("1,2");

        Monitor monitor2 = this.nnet2.getMonitor();
        monitor2.setTrainingPatterns(numberOfTrainingPatterns);
        monitor2.setTotCicles(1);
        monitor2.setLearning(false);

        FileOutputSynapse fileOutputSynapse2 = new FileOutputSynapse();
        fileOutputSynapse2.setFileName("test_out_middle.txt");
        if (this.nnet2 != null) {
            this.nnet2.addInputSynapse(fileInputSynapse2);
            this.nnet2.addOutputSynapse(fileOutputSynapse2);
            System.out.println(this.nnet2.check());
            this.nnet2.start();
            monitor2.Go();
            this.nnet2.join();
        }
    }


    private void initFirstNN() {
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

    private void initSecondNN(){
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

