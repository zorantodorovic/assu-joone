

/**
 * Created by zorantodorovic on 09/01/2017.
 */


import org.joone.engine.*;
import org.joone.engine.learning.TeachingSynapse;
import org.joone.io.FileOutputSynapse;
import org.joone.io.MemoryInputSynapse;
import org.joone.io.MemoryOutputSynapse;
import org.joone.net.NeuralNet;


public class XOR implements NeuralNetListener {
    private NeuralNet nnet = null;
    private MemoryInputSynapse inputSynapse;
    private MemoryInputSynapse desiredOutputSynapse;
    private MemoryOutputSynapse outputSynapse;
    private double[][] inputArray = new double[][]{{0.0D, 0.0D}, {0.0D, 1.0D}, {1.0D, 0.0D}, {1.0D, 1.0D}};
    private double[][] desiredOutputArray = new double[][]{{0.0D}, {1.0D}, {1.0D}, {0.0D}};

    public XOR() {
    }

    public static void main(String[] args) {
        XOR xor = new XOR();
        xor.initNeuralNet();
        xor.train();
    }

    public void train() {
        this.inputSynapse.setInputArray(this.inputArray);
        this.inputSynapse.setAdvancedColumnSelector("1,2");
        this.desiredOutputSynapse.setInputArray(this.desiredOutputArray);
        this.desiredOutputSynapse.setAdvancedColumnSelector("1");
        Monitor monitor = this.nnet.getMonitor();
        monitor.setLearningRate(0.8D);
        monitor.setMomentum(0.3D);
        monitor.setTrainingPatterns(this.inputArray.length);
        monitor.setTotCicles(100000);
        monitor.setLearning(true);
        this.nnet.addNeuralNetListener(this);
        this.nnet.start();
        this.nnet.getMonitor().Go();
        this.nnet.join();
        System.out.println("Network stopped. Last RMSE=" + this.nnet.getMonitor().getGlobalError());
        this.interrogate();
    }

    private void interrogate() {
        Monitor monitor = this.nnet.getMonitor();
        monitor.setTrainingPatterns(4);
        monitor.setTotCicles(1);
        monitor.setLearning(false);
        FileOutputSynapse output = new FileOutputSynapse();
        output.setFileName("xor_out.txt");
        if(this.nnet != null) {
            this.nnet.addOutputSynapse(output);
            System.out.println(this.nnet.check());
            this.nnet.start();
            monitor.Go();
            this.nnet.join();
        }

    }

    protected void initNeuralNet() {
        LinearLayer input = new LinearLayer();
        SigmoidLayer hidden = new SigmoidLayer();
        SigmoidLayer output = new SigmoidLayer();
        input.setRows(2);
        hidden.setRows(3);
        output.setRows(1);
        input.setLayerName("L.input");
        hidden.setLayerName("L.hidden");
        output.setLayerName("L.output");
        FullSynapse synapse_IH = new FullSynapse();
        FullSynapse synapse_HO = new FullSynapse();
        input.addOutputSynapse(synapse_IH);
        hidden.addInputSynapse(synapse_IH);
        hidden.addOutputSynapse(synapse_HO);
        output.addInputSynapse(synapse_HO);
        this.inputSynapse = new MemoryInputSynapse();
        input.addInputSynapse(this.inputSynapse);
        this.outputSynapse = new MemoryOutputSynapse();
        output.addOutputSynapse(this.outputSynapse);
        this.desiredOutputSynapse = new MemoryInputSynapse();
        TeachingSynapse trainer = new TeachingSynapse();
        trainer.setDesired(this.desiredOutputSynapse);
        this.nnet = new NeuralNet();
        this.nnet.addLayer(input, 0);
        this.nnet.addLayer(hidden, 1);
        this.nnet.addLayer(output, 2);
        this.nnet.setTeacher(trainer);
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

