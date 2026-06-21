import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import theme from '../constants/theme';

export default function StartScreen() {
  return (
    <View style={styles.container}>
      <View style={styles.logoBox}>
        <Ionicons name="scan-outline" size={70} color={theme.accent} />
      </View>

      <Text style={styles.title}>DeepSight</Text>
      <Text style={styles.subTitle}>AI 딥페이크 탐지 시스템</Text>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>이미지 속 위조 흔적을 분석합니다</Text>
        <Text style={styles.cardText}>
          얼굴 영역, 피부 질감, 경계 왜곡, 입 주변 패턴을 기반으로 딥페이크 여부를 판단합니다.
        </Text>
      </View>

      <Pressable style={styles.mainBtn} onPress={() => router.push('/upload')}>
        <Text style={styles.mainBtnText}>분석 시작하기</Text>
      </Pressable>

      <Text style={styles.bottomText}>Explainable Deepfake Detection App</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.bg,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  logoBox: {
    width: 130,
    height: 130,
    borderRadius: 35,
    backgroundColor: theme.card,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: theme.border,
    marginBottom: 28,
  },
  title: {
    color: theme.text,
    fontSize: 42,
    fontWeight: '800',
  },
  subTitle: {
    color: theme.subText,
    fontSize: 16,
    marginTop: 8,
    marginBottom: 35,
  },
  card: {
    width: '100%',
    backgroundColor: theme.card,
    borderRadius: 24,
    padding: 22,
    borderWidth: 1,
    borderColor: theme.border,
    marginBottom: 30,
  },
  cardTitle: {
    color: theme.text,
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 10,
  },
  cardText: {
    color: theme.subText,
    fontSize: 14,
    lineHeight: 22,
  },
  mainBtn: {
    width: '100%',
    backgroundColor: theme.primary,
    paddingVertical: 18,
    borderRadius: 18,
    alignItems: 'center',
  },
  mainBtnText: {
    color: theme.text,
    fontSize: 17,
    fontWeight: '700',
  },
  bottomText: {
    color: theme.subText,
    fontSize: 12,
    marginTop: 22,
  },
});