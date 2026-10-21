import React, { useState, useEffect } from 'react';
import {
  SafeAreaView,
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  Platform,
} from 'react-native';
import { StatusBar as ExpoStatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from './src/constants/theme';
import apiService from './src/services/apiService';
import Header from './src/components/Header';
import SettingsModal from './src/components/SettingsModal';
import AssessmentScreen from './src/screens/AssessmentScreen';
import ResultsScreen from './src/screens/ResultsScreen';
import DatasetsScreen from './src/screens/DatasetsScreen';
import HistoryScreen from './src/screens/HistoryScreen';

export default function App() {
  const [activeTab, setActiveTab] = useState('assessment'); // 'assessment', 'results', 'datasets', 'history'
  const [currentResult, setCurrentResult] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [settingsVisible, setSettingsVisible] = useState(false);

  const checkConnection = async () => {
    try {
      await apiService.checkHealth();
      setIsConnected(true);
    } catch (e) {
      setIsConnected(false);
    }
  };

  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleAssessmentComplete = (result) => {
    setCurrentResult(result);
    setActiveTab('results');
  };

  const handleSelectHistoryRecord = (record) => {
    setCurrentResult(record);
    setActiveTab('results');
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ExpoStatusBar style="light" backgroundColor={COLORS.surface} />
      <View style={styles.container}>
        {/* App Header with Auto Connection Status */}
        <Header
          isConnected={isConnected}
          onOpenSettings={() => setSettingsVisible(true)}
        />

        {/* Tab Content */}
        <View style={styles.content}>
          {activeTab === 'assessment' && (
            <AssessmentScreen onAssessmentComplete={handleAssessmentComplete} />
          )}
          {activeTab === 'results' && (
            <ResultsScreen
              result={currentResult}
              onNewAssessment={() => setActiveTab('assessment')}
            />
          )}
          {activeTab === 'datasets' && <DatasetsScreen />}
          {activeTab === 'history' && (
            <HistoryScreen
              onSelectRecord={handleSelectHistoryRecord}
              onNewAssessment={() => setActiveTab('assessment')}
            />
          )}
        </View>

        {/* Bottom Tab Navigation Bar */}
        <View style={styles.tabBar}>
          <TouchableOpacity
            style={styles.tabItem}
            onPress={() => setActiveTab('assessment')}
            activeOpacity={0.7}
          >
            <Ionicons
              name={activeTab === 'assessment' ? 'heart' : 'heart-outline'}
              size={22}
              color={activeTab === 'assessment' ? COLORS.primaryLight : COLORS.textMuted}
            />
            <Text
              style={[
                styles.tabLabel,
                activeTab === 'assessment' && styles.tabLabelActive,
              ]}
            >
              Assess
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.tabItem}
            onPress={() => setActiveTab('results')}
            activeOpacity={0.7}
          >
            <Ionicons
              name={activeTab === 'results' ? 'stats-chart' : 'stats-chart-outline'}
              size={22}
              color={activeTab === 'results' ? COLORS.primaryLight : COLORS.textMuted}
            />
            <Text
              style={[
                styles.tabLabel,
                activeTab === 'results' && styles.tabLabelActive,
              ]}
            >
              Results
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.tabItem}
            onPress={() => setActiveTab('datasets')}
            activeOpacity={0.7}
          >
            <Ionicons
              name={activeTab === 'datasets' ? 'git-network' : 'git-network-outline'}
              size={22}
              color={activeTab === 'datasets' ? COLORS.primaryLight : COLORS.textMuted}
            />
            <Text
              style={[
                styles.tabLabel,
                activeTab === 'datasets' && styles.tabLabelActive,
              ]}
            >
              Hospital Data
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.tabItem}
            onPress={() => setActiveTab('history')}
            activeOpacity={0.7}
          >
            <Ionicons
              name={activeTab === 'history' ? 'time' : 'time-outline'}
              size={22}
              color={activeTab === 'history' ? COLORS.primaryLight : COLORS.textMuted}
            />
            <Text
              style={[
                styles.tabLabel,
                activeTab === 'history' && styles.tabLabelActive,
              ]}
            >
              History
            </Text>
          </TouchableOpacity>
        </View>

        {/* Server & Settings Modal */}
        <SettingsModal
          visible={settingsVisible}
          onClose={() => setSettingsVisible(false)}
          onServerChanged={(connected) => setIsConnected(connected)}
        />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.background,
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 0,
  },
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  content: {
    flex: 1,
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: COLORS.surface,
    borderTopWidth: 1,
    borderTopColor: COLORS.surfaceBorder,
    paddingVertical: 8,
    paddingBottom: Platform.OS === 'ios' ? 24 : 10,
  },
  tabItem: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
  },
  tabLabel: {
    fontSize: 10,
    fontWeight: '600',
    color: COLORS.textMuted,
  },
  tabLabelActive: {
    color: COLORS.primaryLight,
    fontWeight: '800',
  },
});
