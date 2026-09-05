import 'dart:ui';

import 'package:flutter/material.dart';

import 'app_theme.dart';
import 'captures_screen.dart';
import 'home_screen.dart';
import 'phrases_screen.dart';

/// Pantalla principal de la app: navegación por pestañas entre Capturas,
/// Frases y Grabar (cámara en tiempo real).
class MainTabScreen extends StatefulWidget {
  const MainTabScreen({super.key});

  @override
  State<MainTabScreen> createState() => _MainTabScreenState();
}

class _MainTabScreenState extends State<MainTabScreen> {
  int _index = 0;
  final Set<int> _visited = {0};

  static const _screens = [
    CapturesScreen(),
    PhrasesScreen(),
    HomeScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    // Ojo: HomeScreen (pestaña "Grabar") pide la cámara apenas se monta, así
    // que NUNCA la construimos hasta que el usuario realmente toca esa
    // pestaña -- si no, IndexedStack la montaría de entrada y pediría
    // permiso de cámara sin que nadie haya tocado "Grabar".
    final tabs = [
      for (var i = 0; i < _screens.length; i++)
        _visited.contains(i) ? _screens[i] : const SizedBox.shrink(),
    ];

    return Scaffold(
      extendBody: true,
      body: IndexedStack(index: _index, children: tabs),
      bottomNavigationBar: Padding(
        padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(24),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 18, sigmaY: 18),
            child: Container(
              decoration: BoxDecoration(
                color: AppColors.surface.withValues(alpha: 0.86),
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: AppColors.border),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.35),
                    blurRadius: 24,
                    offset: const Offset(0, 10),
                  ),
                ],
              ),
              child: NavigationBar(
                selectedIndex: _index,
                onDestinationSelected: (i) => setState(() {
                  _index = i;
                  _visited.add(i);
                }),
                backgroundColor: Colors.transparent,
                destinations: const [
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
          ),
        ),
      ),
    );
  }
}
