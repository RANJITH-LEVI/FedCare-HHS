import React from 'react';
import {
  Modal,
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../constants/theme';
import { DEFAULT_HOST_IP } from '../constants/api';

export default function InstallModal({ visible, onClose, onInstallPWA, canInstallPWA }) {
  return (
    <Modal visible={visible} animationType="slide" transparent={true} onRequestClose={onClose}>
      <View style={styles.overlay}>
        <View style={styles.modalContent}>
          <View style={styles.header}>
            <View style={styles.headerLeft}>
              <Ionicons name="phone-portrait" size={22} color={COLORS.primaryLight} />
              <Text style={styles.title}>Install as Standalone App</Text>
            </View>
            <TouchableOpacity onPress={onClose}>
              <Ionicons name="close-circle" size={24} color={COLORS.textSecondary} />
            </TouchableOpacity>
          </View>

          <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 20 }}>
            <Text style={styles.introText}>
              You don't need to run this through Expo Go or any third-party app! Here is how to install FedCare
              as an actual standalone app on your phone:
            </Text>

            {/* Method 1: Direct 1-Tap Home Screen Installation */}
            <View style={styles.methodCard}>
              <View style={styles.methodHeader}>
                <View style={styles.methodBadge}>
                  <Text style={styles.methodBadgeText}>METHOD 1 (EASIEST)</Text>
                </View>
                <Text style={styles.methodTitle}>Install Directly to Phone Home Screen</Text>
              </View>

              <Text style={styles.methodDesc}>
                Open your phone's browser and go to:
              </Text>
              <View style={styles.urlBox}>
                <Text style={styles.urlText}>http://{DEFAULT_HOST_IP}:8081</Text>
              </View>

              <View style={styles.stepsList}>
                <View style={styles.stepItem}>
                  <Ionicons name="logo-android" size={18} color={COLORS.riskLow} />
                  <Text style={styles.stepText}>
                    <Text style={{ fontWeight: '700', color: COLORS.text }}>On Android (Chrome/Edge): </Text>
                    Tap the 3 dots menu and select <Text style={{ fontWeight: '700', color: COLORS.primaryLight }}>"Install App"</Text> or <Text style={{ fontWeight: '700', color: COLORS.primaryLight }}>"Add to Home Screen"</Text>. Android automatically generates a native WebAPK that installs into your app drawer.
                  </Text>
                </View>

                <View style={styles.stepItem}>
                  <Ionicons name="logo-apple" size={18} color={COLORS.text} />
                  <Text style={styles.stepText}>
                    <Text style={{ fontWeight: '700', color: COLORS.text }}>On iPhone (Safari): </Text>
                    Tap the <Text style={{ fontWeight: '700', color: COLORS.primaryLight }}>Share icon</Text> at the bottom, then scroll and tap <Text style={{ fontWeight: '700', color: COLORS.primaryLight }}>"Add to Home Screen"</Text>.
                  </Text>
                </View>
              </View>

              {canInstallPWA && (
                <TouchableOpacity style={styles.installNowBtn} onPress={onInstallPWA} activeOpacity={0.8}>
                  <Ionicons name="download" size={18} color={COLORS.white} />
                  <Text style={styles.installNowBtnText}>Install Now to This Device</Text>
                </TouchableOpacity>
              )}
            </View>

            {/* Method 2: Build Standalone Android APK */}
            <View style={styles.methodCard}>
              <View style={styles.methodHeader}>
                <View style={[styles.methodBadge, { backgroundColor: 'rgba(99, 102, 241, 0.2)' }]}>
                  <Text style={[styles.methodBadgeText, { color: COLORS.accent }]}>METHOD 2</Text>
                </View>
                <Text style={styles.methodTitle}>Compile Standalone Android APK (.apk)</Text>
              </View>

              <Text style={styles.methodDesc}>
                We have generated the full native Android project in <Text style={{ fontWeight: '700', color: COLORS.text }}>mobile-app/android</Text>.
                To generate a direct <Text style={{ fontWeight: '700', color: COLORS.primaryLight }}>.apk</Text> installer:
              </Text>

              <View style={styles.codeBlock}>
                <Text style={styles.codeText}>cd mobile-app</Text>
                <Text style={styles.codeText}>npx eas-cli build -p android --profile preview</Text>
              </View>

              <Text style={styles.codeNote}>
                This runs a free cloud build on Expo EAS and outputs a direct link to download the standalone <Text style={{ fontWeight: '700', color: COLORS.text }}>FedCare.apk</Text> file to install on any Android phone!
              </Text>
            </View>
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.75)',
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
    marginBottom: 12,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  title: {
    fontSize: 18,
    fontWeight: '800',
    color: COLORS.text,
  },
  introText: {
    fontSize: 13,
    color: COLORS.textSecondary,
    lineHeight: 18,
    marginBottom: 16,
  },
  methodCard: {
    backgroundColor: COLORS.cardBg,
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
    marginBottom: 14,
  },
  methodHeader: {
    marginBottom: 8,
  },
  methodBadge: {
    alignSelf: 'flex-start',
    backgroundColor: 'rgba(16, 185, 129, 0.15)',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    marginBottom: 4,
  },
  methodBadgeText: {
    fontSize: 10,
    fontWeight: '800',
    color: COLORS.riskLow,
  },
  methodTitle: {
    fontSize: 15,
    fontWeight: '800',
    color: COLORS.text,
  },
  methodDesc: {
    fontSize: 12,
    color: COLORS.textSecondary,
    marginTop: 4,
    lineHeight: 18,
  },
  urlBox: {
    backgroundColor: COLORS.inputBg,
    borderRadius: 8,
    padding: 10,
    marginVertical: 8,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
  },
  urlText: {
    fontSize: 13,
    fontWeight: '700',
    color: COLORS.primaryLight,
    textAlign: 'center',
  },
  stepsList: {
    gap: 10,
    marginVertical: 8,
  },
  stepItem: {
    flexDirection: 'row',
    gap: 10,
  },
  stepText: {
    fontSize: 12,
    color: COLORS.textSecondary,
    flex: 1,
    lineHeight: 17,
  },
  installNowBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: COLORS.riskLow,
    paddingVertical: 12,
    borderRadius: 10,
    marginTop: 10,
  },
  installNowBtnText: {
    fontSize: 13,
    fontWeight: '800',
    color: COLORS.white,
  },
  codeBlock: {
    backgroundColor: COLORS.inputBg,
    borderRadius: 8,
    padding: 10,
    marginVertical: 10,
    borderWidth: 1,
    borderColor: COLORS.surfaceBorder,
  },
  codeText: {
    fontSize: 11,
    fontFamily: 'monospace',
    color: COLORS.primaryLight,
  },
  codeNote: {
    fontSize: 11,
    color: COLORS.textMuted,
    lineHeight: 16,
  },
});
