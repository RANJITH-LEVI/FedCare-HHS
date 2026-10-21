import React, { useState, useEffect } from 'react';
import {
  Modal,
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../constants/theme';
import { getApiUrl, setApiUrl, DEFAULT_HOST_IP } from '../constants/api';
import apiService from '../services/apiService';

export default function SettingsModal({ visible, onClose, onServerChanged }) {
  const [url, setUrl] = useState('');
  const [testing, setTesting] = useState(false);
  const [pingResult, setPingResult] = useState(null);

  useEffect(() => {
    if (visible) {
      getApiUrl().then(setUrl);
      setPingResult(null);
    }
  }, [visible]);

  const handleTest = async () => {
    setTesting(true);
    setPingResult(null);
    const start = Date.now();
    try {
      await setApiUrl(url);
      const res = await apiService.checkHealth();
      const elapsed = Date.now() - start;
      setPingResult({
        success: true,
        ms: elapsed,
        status: res.status || 'online',
        modelReady: res.model_ready,
      });
      if (onServerChanged) onServerChanged(true);
    } catch (err) {
      setPingResult({
        success: false,
        error: err.message,
      });
      if (onServerChanged) onServerChanged(false);
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async () => {
    await setApiUrl(url);
    Alert.alert('Settings Saved', `Backend URL set to:\n${url}`);
    onClose();
  };

  return (
    <Modal visible={visible} animationType="fade" transparent={true} onRequestClose={onClose}>
      <View style={styles.overlay}>
        <View style={styles.content}>
          <View style={styles.header}>
            <View style={styles.headerTitleRow}>
              <Ionicons name="server" size={20} color={COLORS.primaryLight} />
              <Text style={styles.title}>Backend Server Settings</Text>
            </View>
            <TouchableOpacity onPress={onClose}>
              <Ionicons name="close" size={24} color={COLORS.textSecondary} />
            </TouchableOpacity>
          </View>

          <Text style={styles.desc}>
            Configure the FastAPI backend address. If running on a physical phone via Expo Go, use your PC's
            Wi-Fi local IP.
          </Text>

          <Text style={styles.label}>Server Base URL</Text>
          <TextInput
            style={styles.input}
            value={url}
            onChangeText={setUrl}
            placeholder="http://10.185.148.248:8000"
            placeholderTextColor={COLORS.textMuted}
            autoCapitalize="none"
            autoCorrect={false}
          />

          <View style={styles.presetButtonsRow}>
            <TouchableOpacity
              style={styles.presetBtn}
              onPress={() => setUrl(`http://${DEFAULT_HOST_IP}:8000`)}
            >
              <Text style={styles.presetBtnText}>Host Wi-Fi IP</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.presetBtn}
              onPress={() => setUrl('http://localhost:8000')}
            >
              <Text style={styles.presetBtnText}>Localhost</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.presetBtn}
              onPress={() => setUrl('http://10.0.2.2:8000')}
            >
              <Text style={styles.presetBtnText}>Android Emu</Text>
            </TouchableOpacity>
          </View>

          {pingResult && (
            <View
              style={[
                styles.pingBox,
                pingResult.success ? styles.pingSuccess : styles.pingError,
              ]}
            >
              <Ionicons
                name={pingResult.success ? 'checkmark-circle' : 'alert-circle'}
                size={18}
                color={pingResult.success ? COLORS.riskLow : COLORS.riskHigh}
              />
              <Text
                style={[
                  styles.pingText,
                  { color: pingResult.success ? COLORS.riskLow : COLORS.riskHigh },
                ]}
              >
                {pingResult.success
                  ? `Connected (${pingResult.ms}ms) • Status: ${pingResult.status}`
                  : `Failed: ${pingResult.error}`}
              </Text>
            </View>
          )}

          <View style={styles.actions}>
            <TouchableOpacity style={styles.testBtn} onPress={handleTest} disabled={testing}>
              {testing ? (
                <ActivityIndicator size="small" color={COLORS.text} />
              ) : (
                <>
                  <Ionicons name="pulse" size={16} color={COLORS.text} />
                  <Text style={styles.testBtnText}>Test Ping</Text>
                </>
              )}
            </TouchableOpacity>

            <TouchableOpacity style={styles.saveBtn} onPress={handleSave}>
              <Text style={styles.saveBtnText}>Save & Apply</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'center',
    padding: 20,
  },
  content: {
    backgroundColor: COLORS.surface,
    borderRadius: 20,
    padding: 20,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  headerTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  title: {
    fontSize: 16,
    fontWeight: '800',
    color: COLORS.text,
  },
  desc: {
    fontSize: 12,
    color: COLORS.textSecondary,
    lineHeight: 18,
    marginBottom: 16,
  },
  label: {
    fontSize: 11,
    fontWeight: '700',
    color: COLORS.textSecondary,
    marginBottom: 6,
    textTransform: 'uppercase',
  },
  input: {
    backgroundColor: COLORS.inputBg,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: COLORS.text,
    fontSize: 13,
    marginBottom: 10,
  },
  presetButtonsRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 16,
  },
  presetBtn: {
    flex: 1,
    backgroundColor: COLORS.cardBg,
    paddingVertical: 6,
    borderRadius: 8,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
  },
  presetBtnText: {
    fontSize: 10,
    fontWeight: '600',
    color: COLORS.textSecondary,
  },
  pingBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 10,
    borderRadius: 8,
    marginBottom: 16,
  },
  pingSuccess: {
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
  },
  pingError: {
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
  },
  pingText: {
    fontSize: 12,
    fontWeight: '600',
    flex: 1,
  },
  actions: {
    flexDirection: 'row',
    gap: 10,
  },
  testBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    backgroundColor: COLORS.surfaceLight,
    paddingVertical: 12,
    borderRadius: 10,
  },
  testBtnText: {
    color: COLORS.text,
    fontWeight: '700',
    fontSize: 13,
  },
  saveBtn: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: COLORS.primary,
    paddingVertical: 12,
    borderRadius: 10,
  },
  saveBtnText: {
    color: COLORS.white,
    fontWeight: '800',
    fontSize: 13,
  },
});
