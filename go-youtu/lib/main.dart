import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'features/video/presentation/pages/home_shell_page.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(
    const ProviderScope(
      child: GoYoutuApp(),
    ),
  );
}

/// Root widget for the Go-Youtu mobile application (Android & iOS).
class GoYoutuApp extends StatelessWidget {
  const GoYoutuApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Go-Youtu',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.red.shade700),
        useMaterial3: true,
      ),
      // Opens directly into the YouTube video preview card list on startup
      home: const HomeShellPage(),
    );
  }
}
