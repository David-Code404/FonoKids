-- MariaDB dump 10.19  Distrib 10.4.32-MariaDB, for Win64 (AMD64)
--
-- Host: localhost    Database: fonokids
-- ------------------------------------------------------
-- Server version	10.4.32-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Current Database: `fonokids`
--

CREATE DATABASE /*!32312 IF NOT EXISTS*/ `fonokids` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci */;

USE `fonokids`;

--
-- Table structure for table `practice_attempts`
--

DROP TABLE IF EXISTS `practice_attempts`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `practice_attempts` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `class_name` varchar(150) NOT NULL,
  `palabra` varchar(100) NOT NULL,
  `correcta` tinyint(1) NOT NULL,
  `tipo_error` varchar(50) DEFAULT NULL,
  `persona` varchar(100) NOT NULL DEFAULT 'web',
  `prob` float NOT NULL,
  `created_at` datetime NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `idx_palabra` (`palabra`),
  KEY `idx_created_at` (`created_at`)
) ENGINE=InnoDB AUTO_INCREMENT=64 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `practice_attempts`
--

LOCK TABLES `practice_attempts` WRITE;
/*!40000 ALTER TABLE `practice_attempts` DISABLE KEYS */;
INSERT INTO `practice_attempts` VALUES (1,'burro_medias_incorrecto','burro_medias',0,NULL,'web',1,'2026-09-22 18:17:10'),(2,'burro_medias_incorrecto','burro_medias',0,NULL,'web',1,'2026-09-22 18:19:00'),(3,'burro_medias_incorrecto','burro_medias',0,NULL,'web',1,'2026-09-22 18:32:10'),(4,'burro_medias_incorrecto','burro_medias',0,NULL,'web',1,'2026-09-22 18:32:28'),(5,'burro_medias_incorrecto','burro_medias',0,NULL,'web',1,'2026-09-22 18:41:47'),(6,'burro_medias_incorrecto','burro_medias',0,NULL,'web',1,'2026-09-22 18:48:08'),(7,'burro_medias_incorrecto','burro_medias',0,NULL,'web',1,'2026-09-22 19:04:48'),(8,'burro_medias_incorrecto','burro_medias',0,NULL,'web',1,'2026-09-22 19:26:05'),(9,'burro_medias_incorrecto','burro_medias',0,NULL,'web',1,'2026-09-22 19:27:50'),(10,'cama_correcto','cama',1,NULL,'web',1,'2026-09-28 14:24:38'),(11,'carro_correcto','carro',1,NULL,'web',1,'2026-09-22 18:33:11'),(12,'carro_medias_incorrecto','carro_medias',0,NULL,'web',1,'2026-09-22 18:53:23'),(13,'gato_correcto','gato',1,NULL,'web',1,'2026-09-28 17:53:17'),(14,'gorra_correcto','gorra',1,NULL,'web',1,'2026-09-22 18:33:24'),(15,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-22 18:18:04'),(16,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-22 18:29:12'),(17,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-22 18:48:56'),(18,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-22 18:49:07'),(19,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-22 18:52:28'),(20,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-22 18:52:48'),(21,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-22 18:53:01'),(22,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-22 18:53:14'),(23,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-22 19:02:18'),(24,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-22 19:02:44'),(25,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-22 19:02:54'),(26,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-22 19:03:05'),(27,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-22 19:03:17'),(28,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-22 19:03:49'),(29,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-22 19:24:04'),(30,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-22 19:26:54'),(31,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-28 14:20:02'),(32,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-09-28 14:20:17'),(33,'gorra_incorrecto','gorra',0,NULL,'web',1,'2026-10-01 22:28:26'),(34,'perro_correcto','perro',1,NULL,'web',1,'2026-09-26 17:55:01'),(35,'platano_correcto','platano',1,NULL,'web',1,'2026-09-28 17:52:13'),(36,'platano_incorrecto','platano',0,NULL,'web',1,'2026-10-01 22:28:09'),(37,'serpiente_incorrecto','serpiente',0,NULL,'web',1,'2026-09-28 14:24:22');
/*!40000 ALTER TABLE `practice_attempts` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2026-10-05 20:52:25
