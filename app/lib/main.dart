import 'package:flutter/material.dart';

import 'home_screen.dart';

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
      theme: ThemeData(
        colorSchemeSeed: Colors.deepPurple,
        useMaterial3: true,
        brightness: Brightness.dark,
      ),
      home: const HomeScreen(),
    );
  }
}
