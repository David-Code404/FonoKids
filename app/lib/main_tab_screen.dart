import 'package:flutter/material.dart';

import 'app_theme.dart';
import 'captures_screen.dart';
import 'chart_screen.dart';
import 'home_screen.dart';
import 'phrases_screen.dart';

/// Pantalla principal de la app: navegación por pestañas entre Gráficas,
/// Capturas, Frases y Grabar (cámara en tiempo real).
class MainTabScreen extends StatefulWidget {
  const MainTabScreen({super.key});

  @override
  State<MainTabScreen> createState() => _MainTabScreenState();
}

class _MainTabScreenState extends State<MainTabScreen> {
  int _index = 0;

  static const _tabs = [
    ChartScreen(),
    CapturesScreen(),
    PhrasesScreen(),
    HomeScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(index: _index, children: _tabs),
      bottomNavigationBar: Container(
        decoration: const BoxDecoration(
          border: Border(top: BorderSide(color: AppColors.border)),
        ),
        child: NavigationBar(
          selectedIndex: _index,
          onDestinationSelected: (i) => setState(() => _index = i),
          destinations: const [
            NavigationDestination(
              icon: Icon(Icons.bar_chart_outlined),
              selectedIcon: Icon(Icons.bar_chart_rounded),
              label: 'Gráficas',
            ),
            NavigationDestination(
              icon: Icon(Icons.event_note_outlined),
              selectedIcon: Icon(Icons.event_note_rounded),
              label: 'Capturas',
            ),
            NavigationDestination(
              icon: Icon(Icons.record_voice_over_outlined),
              selectedIcon: Icon(Icons.record_voice_over_rounded),
              label: 'Frases',
            ),
            NavigationDestination(
              icon: Icon(Icons.videocam_outlined),
              selectedIcon: Icon(Icons.videocam_rounded),
              label: 'Grabar',
            ),
          ],
        ),
      ),
    );
  }
}
