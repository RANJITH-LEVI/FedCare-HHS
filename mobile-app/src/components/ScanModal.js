import React, { useState } from 'react';
import {
  Modal,
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  Image,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import * as DocumentPicker from 'expo-document-picker';
import { COLORS } from '../constants/theme';
import apiService from '../services/apiService';

// Sample patient profiles for quick demonstration without camera/files
const SAMPLE_PRESETS = [
  {
    name: 'High Risk Cardiac Cohort (Lab #84)',
    desc: 'Age 67, Asymptomatic CP, BP 160, Chol 286, ST Depression 2.6',
    data: {
      age: 67,
      sex: 1,
      cp: 4,
      trestbps: 160,
      chol: 286,
      fbs: 0,
      restecg: 2,
      thalach: 108,
      exang: 1,
      oldpeak: 2.6,
      slope: 2,
      ca: 3,
      thal: 7,
    },
  },
  {
    name: 'Typical Angina Patient (Lab #41)',
    desc: 'Age 53, Anginal Pain, BP 140, Chol 203, ST Slope 3.0',
    data: {
      age: 53,
      sex: 1,
      cp: 1,
      trestbps: 140,
      chol: 203,
      fbs: 1,
      restecg: 2,
      thalach: 155,
      exang: 1,
      oldpeak: 3.1,
      slope: 3,
      ca: 0,
      thal: 7,
    },
  },
  {
    name: 'Healthy Athletic Adult (Lab #12)',
    desc: 'Age 41, Atypical, BP 130, Chol 204, High MaxHR 172',
    data: {
      age: 41,
      sex: 0,
      cp: 2,
      trestbps: 130,
      chol: 204,
      fbs: 0,
      restecg: 2,
      thalach: 172,
      exang: 0,
      oldpeak: 1.4,
      slope: 1,
      ca: 0,
      thal: 3,
    },
  },
];

export default function ScanModal({ visible, onClose, onDataExtracted }) {
  const [loading, setLoading] = useState(false);
  const [statusText, setStatusText] = useState('');
  const [previewUri, setPreviewUri] = useState(null);

  const handleScanFile = async (uri, mimeType, fileName) => {
    setLoading(true);
    setStatusText('Gemini Vision AI analyzing medical report...');
    try {
      const response = await apiService.scanDocument(uri, mimeType, fileName);
      if (response && response.extracted_features) {
        setStatusText('Extraction complete! Populating form...');
        setTimeout(() => {
          setLoading(false);
          setPreviewUri(null);
          onClose();
          onDataExtracted(response.extracted_features);
        }, 600);
      } else {
        throw new Error('No clinical features detected in document.');
      }
    } catch (err) {
      setLoading(false);
      Alert.alert('Scan Failed', err.message || 'Could not parse document. Try another file or quick preset.');
    }
  };

  const handleCamera = async () => {
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Camera permission is required to photograph reports.');
        return;
      }

      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        quality: 0.8,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        const asset = result.assets[0];
        setPreviewUri(asset.uri);
        await handleScanFile(asset.uri, asset.mimeType || 'image/jpeg', 'camera_report.jpg');
      }
    } catch (e) {
      Alert.alert('Camera Error', e.message);
    }
  };

  const handleGallery = async () => {
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Photo library permission is required.');
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        quality: 0.8,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        const asset = result.assets[0];
        setPreviewUri(asset.uri);
        await handleScanFile(asset.uri, asset.mimeType || 'image/jpeg', 'gallery_report.jpg');
      }
    } catch (e) {
      Alert.alert('Gallery Error', e.message);
    }
  };

  const handleDocument = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ['application/pdf', 'image/*'],
        copyToCacheDirectory: true,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        const asset = result.assets[0];
        setPreviewUri(asset.mimeType?.startsWith('image') ? asset.uri : null);
        await handleScanFile(asset.uri, asset.mimeType || 'application/pdf', asset.name || 'report.pdf');
      }
    } catch (e) {
      Alert.alert('File Picker Error', e.message);
    }
  };

  const handlePresetSelect = (preset) => {
    setLoading(true);
    setStatusText(`Loading sample ${preset.name}...`);
    setTimeout(() => {
      setLoading(false);
      onClose();
      onDataExtracted(preset.data);
    }, 400);
  };

  return (
    <Modal visible={visible} animationType="slide" transparent={true} onRequestClose={onClose}>
      <View style={styles.overlay}>
        <View style={styles.modalContent}>
          <View style={styles.header}>
            <View style={styles.headerTitleRow}>
              <Ionicons name="document-text" size={22} color={COLORS.primaryLight} />
              <Text style={styles.title}>Scan Clinical Document</Text>
            </View>
            <TouchableOpacity onPress={onClose} disabled={loading}>
              <Ionicons name="close-circle" size={26} color={COLORS.textSecondary} />
            </TouchableOpacity>
          </View>

          <Text style={styles.description}>
            Capture or upload a patient lab report, discharge summary, or ECG slip. Gemini Vision
            will automatically extract the 13 clinical biomarkers.
          </Text>

          {loading ? (
            <View style={styles.loadingContainer}>
              {previewUri && (
                <Image source={{ uri: previewUri }} style={styles.previewThumb} resizeMode="cover" />
              )}
              <ActivityIndicator size="large" color={COLORS.primaryLight} style={{ marginTop: 16 }} />
              <Text style={styles.loadingText}>{statusText}</Text>
              <Text style={styles.loadingSubtext}>Extracting blood pressure, cholesterol, ECG, ST depression...</Text>
            </View>
          ) : (
            <>
              {/* Action Buttons */}
              <View style={styles.buttonGrid}>
                <TouchableOpacity style={styles.actionBtn} onPress={handleCamera} activeOpacity={0.8}>
                  <View style={[styles.iconWrap, { backgroundColor: 'rgba(56, 189, 248, 0.15)' }]}>
                    <Ionicons name="camera" size={24} color={COLORS.primaryLight} />
                  </View>
                  <Text style={styles.btnTitle}>Take Photo</Text>
                  <Text style={styles.btnDesc}>Use phone camera</Text>
                </TouchableOpacity>

                <TouchableOpacity style={styles.actionBtn} onPress={handleGallery} activeOpacity={0.8}>
                  <View style={[styles.iconWrap, { backgroundColor: 'rgba(99, 102, 241, 0.15)' }]}>
                    <Ionicons name="images" size={24} color={COLORS.accent} />
                  </View>
                  <Text style={styles.btnTitle}>Photo Library</Text>
                  <Text style={styles.btnDesc}>Select from gallery</Text>
                </TouchableOpacity>

                <TouchableOpacity style={styles.actionBtn} onPress={handleDocument} activeOpacity={0.8}>
                  <View style={[styles.iconWrap, { backgroundColor: 'rgba(16, 185, 129, 0.15)' }]}>
                    <Ionicons name="folder-open" size={24} color={COLORS.riskLow} />
                  </View>
                  <Text style={styles.btnTitle}>PDF / File</Text>
                  <Text style={styles.btnDesc}>Select PDF report</Text>
                </TouchableOpacity>
              </View>

              {/* Sample Presets Divider */}
              <View style={styles.dividerRow}>
                <View style={styles.dividerLine} />
                <Text style={styles.dividerText}>OR QUICK DEMO LAB PRESETS</Text>
                <View style={styles.dividerLine} />
              </View>

              {/* Quick Presets */}
              <View style={styles.presetList}>
                {SAMPLE_PRESETS.map((p, idx) => (
                  <TouchableOpacity
                    key={idx}
                    style={styles.presetCard}
                    onPress={() => handlePresetSelect(p)}
                    activeOpacity={0.7}
                  >
                    <View style={styles.presetLeft}>
                      <Ionicons name="flask" size={18} color={COLORS.primaryLight} />
                      <View style={{ flex: 1 }}>
                        <Text style={styles.presetName}>{p.name}</Text>
                        <Text style={styles.presetDesc}>{p.desc}</Text>
                      </View>
                    </View>
                    <Ionicons name="chevron-forward" size={16} color={COLORS.textSecondary} />
                  </TouchableOpacity>
                ))}
              </View>
            </>
          )}
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.75)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: COLORS.surface,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    padding: 20,
    maxHeight: '90%',
    borderTopWidth: 1,
    borderTopColor: COLORS.surfaceBorder,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  headerTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  title: {
    fontSize: 18,
    fontWeight: '800',
    color: COLORS.text,
  },
  description: {
    fontSize: 13,
    color: COLORS.textSecondary,
    lineHeight: 18,
    marginBottom: 20,
  },
  buttonGrid: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 20,
  },
  actionBtn: {
    flex: 1,
    backgroundColor: COLORS.cardBg,
    borderRadius: 14,
    padding: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
  },
  iconWrap: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  btnTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: COLORS.text,
  },
  btnDesc: {
    fontSize: 10,
    color: COLORS.textMuted,
    marginTop: 2,
  },
  dividerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginVertical: 12,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: COLORS.surfaceBorder,
  },
  dividerText: {
    fontSize: 10,
    color: COLORS.textMuted,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  presetList: {
    gap: 8,
    marginBottom: 10,
  },
  presetCard: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: COLORS.cardBg,
    padding: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
  },
  presetLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    flex: 1,
  },
  presetName: {
    fontSize: 13,
    fontWeight: '700',
    color: COLORS.text,
  },
  presetDesc: {
    fontSize: 11,
    color: COLORS.textSecondary,
    marginTop: 2,
  },
  loadingContainer: {
    alignItems: 'center',
    paddingVertical: 30,
  },
  previewThumb: {
    width: 120,
    height: 120,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: COLORS.primaryLight,
  },
  loadingText: {
    fontSize: 15,
    fontWeight: '700',
    color: COLORS.text,
    marginTop: 12,
  },
  loadingSubtext: {
    fontSize: 12,
    color: COLORS.textSecondary,
    marginTop: 4,
    textAlign: 'center',
  },
});
