package ch.zhaw.deeplearningjava.playground;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class SentimentAnalysis {

    private static final Logger logger = LoggerFactory.getLogger(SentimentAnalysis.class);
    Predictor<String, Classifications> predictor;

    public SentimentAnalysis() {
        try {
            Criteria<String, Classifications> criteria = Criteria.builder()
                    .optApplication(Application.NLP.SENTIMENT_ANALYSIS)
                    .setTypes(String.class, Classifications.class)
                    // This model was traced on CPU and can only run on CPU
                    .optDevice(Device.cpu())
                    .optProgress(new ProgressBar())
                    .build();
            ZooModel<String, Classifications> model = criteria.loadModel();
            predictor = model.newPredictor();
        } catch (Exception e) {
            logger.error("Cannot build Predictor", e);
        }
    }

    public Classifications predict(String input) throws Exception {
        // String input = "I like DJL. DJL is the best DL framework!";
        logger.info("input Sentence: {}", input);
        return predictor.predict(input);
    }
}
