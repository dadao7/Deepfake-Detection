import * as Print from 'expo-print';
import { router, useLocalSearchParams } from 'expo-router';
import * as Sharing from 'expo-sharing';
import { Alert, Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import theme from '../constants/theme';

export default function ReportScreen() {
  const {
    imageUri,
    label,
    resultTitle,
    fakeProbability,
    baselineProbability,
    detailProbability,
    summary,
    reasons,
    llmExplanation,
  } = useLocalSearchParams<{
    imageUri: string;
    label: string;
    resultTitle: string;
    fakeProbability: string;
    baselineProbability: string;
    detailProbability: string;
    summary: string;
    reasons: string;
    llmExplanation: string;
  }>();

  const isFake = label === 'fake';
  const result = isFake ? 'FAKE' : 'REAL';

  const parsedReasons: string[] = reasons ? JSON.parse(reasons) : [];

  const percent = (value?: string) => {
    if (!value) return '0.0%';
    return `${Number(value).toFixed(1)}%`;
  };

  const downloadPDF = async () => {
    try {
      const reasonHtml = parsedReasons
        .map((item) => `<li>${item}</li>`)
        .join('');

      const html = `
        <html>
          <head>
            <meta charset="utf-8" />
            <style>
              body {
                font-family: Arial, sans-serif;
                padding: 24px;
                color: #222;
              }
              h1 {
                font-size: 28px;
                margin-bottom: 20px;
              }
              img {
                width: 100%;
                max-height: 320px;
                object-fit: contain;
                border-radius: 16px;
                margin-bottom: 20px;
              }
              .card {
                border: 1px solid #ddd;
                border-radius: 16px;
                padding: 18px;
                margin-bottom: 16px;
              }
              .title {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
              }
              .result {
                font-size: 30px;
                font-weight: bold;
                color: ${isFake ? '#ff4d4d' : '#2ecc71'};
              }
              p, li {
                font-size: 14px;
                line-height: 1.7;
              }
            </style>
          </head>
          <body>
            <h1>AI 분석 리포트</h1>

            <img src="${imageUri}" />

            <div class="card">
              <div class="title">최종 판단</div>
              <div class="result">${result}</div>
              <p>딥페이크 확률: ${percent(fakeProbability)}</p>
              <p>전체 영역 기반 딥페이크 확률: ${percent(baselineProbability)}</p>
              <p>세부 영역 기반 딥페이크 확률: ${percent(detailProbability)}</p>
            </div>

            <div class="card">
              <div class="title">분석 요약</div>
              <p>${summary || '분석 요약을 불러오지 못했습니다.'}</p>
            </div>

            <div class="card">
              <div class="title">AI 상세 설명</div>
              <p>${llmExplanation || 'AI 상세 설명을 불러오지 못했습니다.'}</p>
            </div>
          </body>
        </html>
      `;

      const { uri } = await Print.printToFileAsync({ html });

      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(uri);
      } else {
        Alert.alert('PDF 생성 완료', `PDF 파일이 생성되었습니다.\n${uri}`);
      }
    } catch (error) {
      Alert.alert('오류', 'PDF 다운로드 중 문제가 발생했습니다.');
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>AI 분석 리포트</Text>

      <Image source={{ uri: imageUri }} style={styles.image} />

      <View style={styles.card}>
        <Text style={styles.cardTitle}>최종 판단</Text>

        <Text style={[styles.result, { color: isFake ? theme.danger : theme.success }]}>
          {result}
        </Text>

        <Text style={styles.text}>딥페이크 확률: {percent(fakeProbability)}</Text>
        <Text style={styles.text}>전체 영역 기반 딥페이크 확률: {percent(baselineProbability)}</Text>
        <Text style={styles.text}>세부 영역 기반 딥페이크 확률: {percent(detailProbability)}</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>분석 요약</Text>
        <Text style={styles.text}>{summary || '분석 요약을 불러오지 못했습니다.'}</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>AI 상세 설명</Text>
        <Text style={styles.text}>
          {llmExplanation || 'AI 상세 설명을 불러오지 못했습니다.'}
        </Text>
      </View>

      <Pressable style={styles.pdfBtn} onPress={downloadPDF}>
        <Text style={styles.pdfBtnText}>PDF 다운로드하기</Text>
      </Pressable>

      <Pressable style={styles.btn} onPress={() => router.replace('/')}>
        <Text style={styles.btnText}>새 분석 시작하기</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.bg,
  },
  content: {
    padding: 22,
    paddingBottom: 40,
  },
  title: {
    color: theme.text,
    fontSize: 28,
    fontWeight: '800',
    marginBottom: 20,
  },
  image: {
    width: '100%',
    height: 260,
    borderRadius: 24,
    marginBottom: 22,
  },
  card: {
    backgroundColor: theme.card,
    borderRadius: 24,
    padding: 22,
    borderWidth: 1,
    borderColor: theme.border,
    marginBottom: 18,
  },
  cardTitle: {
    color: theme.text,
    fontSize: 18,
    fontWeight: '800',
    marginBottom: 14,
  },
  result: {
    fontSize: 34,
    fontWeight: '900',
    marginBottom: 8,
  },
  text: {
    color: theme.subText,
    fontSize: 15,
    lineHeight: 24,
  },
  pdfBtn: {
    backgroundColor: theme.card,
    paddingVertical: 17,
    borderRadius: 18,
    alignItems: 'center',
    marginTop: 8,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: theme.primary,
  },
  pdfBtnText: {
    color: theme.primary,
    fontSize: 16,
    fontWeight: '800',
  },
  btn: {
    backgroundColor: theme.primary,
    paddingVertical: 17,
    borderRadius: 18,
    alignItems: 'center',
  },
  btnText: {
    color: theme.text,
    fontSize: 16,
    fontWeight: '700',
  },
});