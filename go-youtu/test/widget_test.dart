import 'package:flutter_test/flutter_test.dart';
import 'package:go_youtu/features/video/presentation/providers/video_providers.dart';

void main() {
  test('TopNavigationTab 頂部六個清單順序與標籤正確 (含排除頻道)', () {
    expect(
      TopNavigationTab.values.map((e) => e.label).toList(),
      equals(<String>[
        '影片清單',
        '已下載過清單',
        '訂閱清單',
        '下載佇列清單',
        '已看過清單',
        '排除頻道',
      ]),
    );
  });
}
