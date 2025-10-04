"""
CNN Models for Skin Lesion Classification
Supports multiple architectures with binary classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
from tensorflow.keras.metrics import Precision, Recall, AUC

class SkinLesionCNN:
    """
    CNN model class for skin lesion classification.
    Supports multiple architectures with transfer learning.
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2, 
                 architecture='resnet50', pretrained=True, learning_rate=0.0001,
                 pos_weight=None, binary_classification=True):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes (2 for binary)
            architecture: Model architecture
            pretrained: Whether to use pretrained ImageNet weights
            learning_rate: Initial learning rate for optimizer
            pos_weight: Positive class weight for binary classification
            binary_classification: Whether this is binary classification
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture = architecture.lower()
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.pos_weight = pos_weight
        self.binary_classification = binary_classification
        self.model = None
        self.base_model = None
    
    def build_simple_cnn(self, dropout_rate=0.5):
        """Build a simple CNN for testing (not for production)."""
        self.model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout_rate),
            
            # Output layer (binary classification)
            layers.Dense(1, dtype='float32', activation='sigmoid')
        ])
        
        self._compile_model()
        return self.model
    
    def build_resnet50(self, pretrained=True, fine_tune_layers=50):
        """Build ResNet50 with transfer learning."""
        weights = 'imagenet' if pretrained else None
        
        # Load base model
        self.base_model = ResNet50(
            weights=weights,
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model initially
        self.base_model.trainable = False
        
        # Build full model
        inputs = keras.Input(shape=self.input_shape)
        x = self.base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer (binary classification with float32)
        outputs = layers.Dense(1, dtype='float32', activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        self._compile_model()
        return self.model
    
    def build_vgg16(self, pretrained=True):
        """Build VGG16 with transfer learning."""
        weights = 'imagenet' if pretrained else None
        
        # Load base model
        self.base_model = VGG16(
            weights=weights,
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model initially
        self.base_model.trainable = False
        
        # Build full model
        inputs = keras.Input(shape=self.input_shape)
        x = self.base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(1, dtype='float32', activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        self._compile_model()
        return self.model
    
    def build_inception_v3(self, pretrained=True):
        """Build InceptionV3 with transfer learning."""
        weights = 'imagenet' if pretrained else None
        
        # Load base model
        self.base_model = InceptionV3(
            weights=weights,
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model initially
        self.base_model.trainable = False
        
        # Build full model
        inputs = keras.Input(shape=self.input_shape)
        x = self.base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(1, dtype='float32', activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        self._compile_model()
        return self.model
    
    def build_efficientnet(self, pretrained=True):
        """Build EfficientNet with transfer learning (v1/v2 compatible)."""
        weights = 'imagenet' if pretrained else None
        
        # Try EfficientNetB0 (v1) first, fall back to v2
        try:
            from tensorflow.keras.applications import EfficientNetB0
            self.base_model = EfficientNetB0(
                weights=weights,
                include_top=False,
                input_shape=self.input_shape
            )
        except (ImportError, AttributeError):
            try:
                from tensorflow.keras.applications import EfficientNetV2B0
                self.base_model = EfficientNetV2B0(
                    weights=weights,
                    include_top=False,
                    input_shape=self.input_shape
                )
            except Exception as e:
                raise ImportError(f"Could not import EfficientNet: {e}")
        
        # Freeze base model initially
        self.base_model.trainable = False
        
        # Build full model
        inputs = keras.Input(shape=self.input_shape)
        x = self.base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(1, dtype='float32', activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        self._compile_model()
        return self.model
    
    def _compile_model(self):
        """Compile the model with optimizer, loss, and metrics."""
        
        # Try AdamW first, fall back to Adam
        try:
            optimizer = keras.optimizers.AdamW(
                learning_rate=self.learning_rate,
                weight_decay=1e-4
            )
        except (AttributeError, TypeError):
            try:
                # Try tensorflow_addons
                import tensorflow_addons as tfa
                optimizer = tfa.optimizers.AdamW(
                    learning_rate=self.learning_rate,
                    weight_decay=1e-4
                )
            except ImportError:
                # Fall back to standard Adam
                optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
                print("AdamW not available, using Adam optimizer")
        
        # Binary classification metrics
        metrics = [
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc_roc', curve='ROC'),
            AUC(name='pr_auc', curve='PR')
        ]
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=metrics
        )
    
    def unfreeze_base_model(self, num_layers=30, learning_rate=0.00001):
        """
        Unfreeze the last N layers of the base model for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the end
            learning_rate: Learning rate for fine-tuning (should be lower)
        
        Returns:
            success: True if unfreezing was successful
        """
        if self.base_model is None:
            print("No base model to unfreeze (using simple_cnn?)")
            return False
        
        # Unfreeze the base model
        self.base_model.trainable = True
        
        # Freeze all layers except the last num_layers
        total_layers = len(self.base_model.layers)
        freeze_until = max(0, total_layers - num_layers)
        
        for i, layer in enumerate(self.base_model.layers):
            if i < freeze_until:
                layer.trainable = False
            else:
                layer.trainable = True
        
        print(f"\nUnfreezing last {num_layers} layers of {self.architecture}")
        print(f"Total base layers: {total_layers}")
        print(f"Trainable layers: {sum([1 for l in self.base_model.layers if l.trainable])}")
        
        # Recompile with lower learning rate
        try:
            optimizer = keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=1e-4
            )
        except (AttributeError, TypeError):
            try:
                import tensorflow_addons as tfa
                optimizer = tfa.optimizers.AdamW(
                    learning_rate=learning_rate,
                    weight_decay=1e-4
                )
            except ImportError:
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        metrics = [
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc_roc', curve='ROC'),
            AUC(name='pr_auc', curve='PR')
        ]
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=metrics
        )
        
        print(f"Recompiled with learning_rate={learning_rate}")
        return True
    
    def get_model(self):
        """Return the compiled model."""
        return self.model
    
    def get_model_info(self):
        """Get model information including parameter counts."""
        if self.model is None:
            return {
                'total_parameters': 0,
                'trainable_parameters': 0,
                'non_trainable_parameters': 0
            }
        
        trainable_count = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_count = sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
        
        return {
            'total_parameters': trainable_count + non_trainable_count,
            'trainable_parameters': trainable_count,
            'non_trainable_parameters': non_trainable_count
        }
    
    def summary(self):
        """Print model summary."""
        if self.model:
            return self.model.summary()
        else:
            print("Model not built yet. Call a build_* method first.")


def create_ensemble_model(models_list, input_shape=(224, 224, 3)):
    """
    Create an ensemble model from multiple trained models.
    
    Args:
        models_list: List of trained Keras models
        input_shape: Input shape for the ensemble
        
    Returns:
        Ensemble model that averages predictions
    """
    if not models_list:
        raise ValueError("models_list cannot be empty")
    
    # Create input layer
    inputs = keras.Input(shape=input_shape)
    
    # Get predictions from each model
    predictions = []
    for model in models_list:
        # Set models to inference mode
        model.trainable = False
        pred = model(inputs)
        predictions.append(pred)
    
    # Average predictions
    if len(predictions) > 1:
        averaged = layers.Average()(predictions)
    else:
        averaged = predictions[0]
    
    # Create ensemble model
    ensemble_model = keras.Model(inputs=inputs, outputs=averaged)
    
    return ensemble_model

