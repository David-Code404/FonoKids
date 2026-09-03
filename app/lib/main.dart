import 'package:flutter/material.dart';

import 'app_theme.dart';
import 'main_tab_screen.dart';

void main() {
  runApp(const SpeakShadowApp());
}

class SpeakShadowApp extends StatelessWidget {
  const SpeakShadowApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SpeakShadow',
      debugShowCheckedModeBanner: false,
      theme: buildAppTheme(),
      home: const MainTabScreen(),
    );
  }
}
