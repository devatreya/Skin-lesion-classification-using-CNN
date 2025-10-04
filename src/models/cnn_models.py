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
                 architecture='resnet50', pretrained=True):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes (2 for binary)
            architecture: Model architecture ('resnet50', 'vgg16', 'inception_v3', 
                         'efficientnet', 'simple_cnn')
            pretrained: Whether to use pretrained ImageNet weights
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture = architecture.lower()
        self.pretrained = pretrained
        self.model = None
        self.base_model = None
        
        # Build the model
        self._build_model()
    
    def _build_model(self):
        """Build the CNN model based on specified architecture."""
        
        if self.architecture == 'simple_cnn':
            self.model = self._build_simple_cnn()
        elif self.architecture == 'resnet50':
            self.model = self._build_resnet50()
        elif self.architecture == 'vgg16':
            self.model = self._build_vgg16()
        elif self.architecture == 'inception_v3':
            self.model = self._build_inception_v3()
        elif self.architecture == 'efficientnet':
            self.model = self._build_efficientnet()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        # Compile the model
        self._compile_model()
    
    def _build_simple_cnn(self):
        """Build a simple CNN for testing (not for production)."""
        model = models.Sequential([
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
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer (binary classification)
            layers.Dense(1, dtype='float32', activation='sigmoid')
        ])
        
        return model
    
    def _build_resnet50(self):
        """Build ResNet50 with transfer learning."""
        weights = 'imagenet' if self.pretrained else None
        
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
        
        model = keras.Model(inputs, outputs)
        return model
    
    def _build_vgg16(self):
        """Build VGG16 with transfer learning."""
        weights = 'imagenet' if self.pretrained else None
        
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
        
        model = keras.Model(inputs, outputs)
        return model
    
    def _build_inception_v3(self):
        """Build InceptionV3 with transfer learning."""
        weights = 'imagenet' if self.pretrained else None
        
        # InceptionV3 requires input shape of at least 75x75
        if self.input_shape[0] < 75 or self.input_shape[1] < 75:
            raise ValueError("InceptionV3 requires input shape >= 75x75")
        
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
        
        model = keras.Model(inputs, outputs)
        return model
    
    def _build_efficientnet(self):
        """Build EfficientNet with transfer learning (v1/v2 compatible)."""
        weights = 'imagenet' if self.pretrained else None
        
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
        
        model = keras.Model(inputs, outputs)
        return model
    
    def _compile_model(self):
        """Compile the model with optimizer, loss, and metrics."""
        
        # Try AdamW first, fall back to Adam
        try:
            optimizer = keras.optimizers.AdamW(
                learning_rate=0.0001,
                weight_decay=1e-4
            )
        except (AttributeError, TypeError):
            try:
                # Try tensorflow_addons
                import tensorflow_addons as tfa
                optimizer = tfa.optimizers.AdamW(
                    learning_rate=0.0001,
                    weight_decay=1e-4
                )
            except ImportError:
                # Fall back to standard Adam
                optimizer = keras.optimizers.Adam(learning_rate=0.0001)
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
        """
        if self.base_model is None:
            print("No base model to unfreeze (using simple_cnn?)")
            return
        
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
    
    def get_model(self):
        """Return the compiled model."""
        return self.model
    
    def summary(self):
        """Print model summary."""
        return self.model.summary()


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

