import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import json
import os
from tqdm import tqdm  # Pour la barre de progression
import itertools

def compter_marches(chemin_json):
    try:
        with open(chemin_json, 'r') as f:
            donnees = json.load(f)
        
        if 'shapes' in donnees:
            # Filtrer les formes qui ont le label 'marche'
            marches = [shape for shape in donnees['shapes'] if shape.get('label') == 'marche']
            nombre_marches = len(marches)
            return nombre_marches
        else:
            print(f"Format JSON non reconnu dans {chemin_json}: clé 'shapes' non trouvée")
            return 0
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier JSON {chemin_json}: {e}")
        return 0

def detect_stairs_with_params(image_path, params, show_results=False):
    """
    Détecte les marches d'escalier avec des paramètres spécifiés
    """
    # Extraction des paramètres
    blur_kernel = params.get('blur_kernel', (5, 5))
    canny_low = params.get('canny_low', 50)
    canny_high = params.get('canny_high', 150)
    hough_threshold = params.get('hough_threshold', 200)
    min_line_length = params.get('min_line_length', 80)
    max_line_gap = params.get('max_line_gap', 10)
    horizontal_angle_threshold = params.get('horizontal_angle_threshold', 5)
    dbscan_eps = params.get('dbscan_eps', 20)
    dbscan_min_samples = params.get('dbscan_min_samples', 2)
    dominant_distance_factor = params.get('dominant_distance_factor', 0.5)
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Impossible de charger l'image {image_path}")
        return 0
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    
    # Détection des bords avec Canny
    edges = cv2.Canny(blurred, canny_low, canny_high, apertureSize=3)
    
    # Détection des lignes avec la transformée de Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_threshold, 
                           minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    # Préparation des images de sortie
    out_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    control = np.copy(out_img)
    
    # Liste pour stocker les hauteurs des marches détectées
    y_keeper_for_lines = []
    
    # Filtrer les lignes détectées
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if abs(angle) < horizontal_angle_threshold:  # Ne garder que les lignes horizontales
                y_avg = int(np.mean([y1, y2]))  # Moyenne pour éviter du bruit
                cv2.line(control, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
                y_keeper_for_lines.append(y_avg)
    
    # Clustering des hauteurs des lignes détectées
    if y_keeper_for_lines:
        y_keeper_for_lines = np.array(y_keeper_for_lines).reshape(-1, 1)
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(y_keeper_for_lines)
        unique_clusters = set(db.labels_) - {-1}  # Exclure les bruits
        filtered_y = [int(np.mean(y_keeper_for_lines[db.labels_ == c])) for c in unique_clusters]
    else:
        filtered_y = []
    
    # Trier les hauteurs des lignes détectées de haut en bas
    filtered_y = sorted(filtered_y)
    
    # Calculer les écarts entre les lignes détectées
    if len(filtered_y) > 1:
        distances = np.diff(filtered_y)  # Liste des écarts entre marches
        dominant_distance = np.median(distances)  # Distance dominante entre marches
    else:
        dominant_distance = 30  # Valeur par défaut si une seule marche est détectée
    
    # Filtrer les marches en fonction de la distance dominante
    final_y = []
    prev_y = -999  # Valeur initiale éloignée
    
    for y in filtered_y:
        if abs(y - prev_y) > dominant_distance_factor * dominant_distance:  # On garde une flexibilité
            final_y.append(y)
            prev_y = y
    
    # Nombre de marches détectées
    stair_counter = len(final_y)
    
    # Dessiner les marches détectées si affichage demandé
    if show_results:
        for y in final_y:
            cv2.line(out_img, (0, y), (image.shape[1], y), (0, 255, 0), 3, cv2.LINE_AA)
        
        # Ajouter le texte indiquant le nombre de marches détectées
        cv2.putText(out_img, f"Nombre de marches: {stair_counter}", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Affichage des résultats
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Image Originale")
        ax[0].axis("off")
        
        ax[1].imshow(control)
        ax[1].set_title("Lignes détectées")
        ax[1].axis("off")
        
        ax[2].imshow(out_img)
        ax[2].set_title("Marches détectées")
        ax[2].axis("off")
        
        plt.show()
    
    return stair_counter

def find_best_params_for_image(image_path, json_path, param_ranges):
    """
    Trouve les meilleurs paramètres pour une image spécifique
    """
    target_steps = compter_marches(json_path)
    if target_steps == 0:
        print(f"Pas de marches définies pour {image_path}")
        return None, 0
    
    best_params = None
    best_error = float('inf')
    
    # Réduction de l'espace de recherche: sélection de combinaisons plus intelligentes
    # Plutôt que de tester toutes les combinaisons, on sélectionne un sous-ensemble plus intelligent
    param_combinations = []
    
    # Première phase: tester des combinaisons éloignées pour explorer l'espace de recherche
    for blur_kernel in param_ranges['blur_kernel']:
        for canny_low in [param_ranges['canny_low'][0], param_ranges['canny_low'][-1]]:
            for canny_high in [param_ranges['canny_high'][0], param_ranges['canny_high'][-1]]:
                for hough_threshold in [param_ranges['hough_threshold'][1]]:
                    for min_line_length in [param_ranges['min_line_length'][1]]:
                        for max_line_gap in [param_ranges['max_line_gap'][1]]:
                            for horizontal_angle_threshold in [param_ranges['horizontal_angle_threshold'][1]]:
                                for dbscan_eps in [param_ranges['dbscan_eps'][1]]:
                                    for dbscan_min_samples in param_ranges['dbscan_min_samples']:
                                        for dominant_distance_factor in [param_ranges['dominant_distance_factor'][1]]:
                                            params = {
                                                'blur_kernel': blur_kernel,
                                                'canny_low': canny_low,
                                                'canny_high': canny_high,
                                                'hough_threshold': hough_threshold,
                                                'min_line_length': min_line_length,
                                                'max_line_gap': max_line_gap,
                                                'horizontal_angle_threshold': horizontal_angle_threshold,
                                                'dbscan_eps': dbscan_eps,
                                                'dbscan_min_samples': dbscan_min_samples,
                                                'dominant_distance_factor': dominant_distance_factor
                                            }
                                            param_combinations.append(params)
    
    # Tester chaque combinaison de la première phase
    for params in tqdm(param_combinations, desc=f"Optimisation phase 1 pour {os.path.basename(image_path)}"):
        detected_steps = detect_stairs_with_params(image_path, params)
        error = abs(detected_steps - target_steps)
        
        if error < best_error:
            best_error = error
            best_params = params
            
            # Si parfaite correspondance, on arrête
            if error == 0:
                return best_params, best_error
    
    # Deuxième phase: exploration plus fine autour des meilleurs paramètres trouvés
    if best_params is not None:
        fine_tune_ranges = {}
        for param_name, param_values in param_ranges.items():
            if param_name == 'blur_kernel':
                current_index = param_values.index(best_params[param_name])
                fine_tune_ranges[param_name] = [best_params[param_name]]
                if current_index > 0:
                    fine_tune_ranges[param_name].append(param_values[current_index-1])
                if current_index < len(param_values)-1:
                    fine_tune_ranges[param_name].append(param_values[current_index+1])
            else:
                current_value = best_params[param_name]
                if param_name in ['canny_low', 'canny_high', 'hough_threshold', 'min_line_length']:
                    fine_tune_ranges[param_name] = [max(current_value - 20, 10), current_value, min(current_value + 20, 300)]
                elif param_name in ['max_line_gap', 'dbscan_eps']:
                    fine_tune_ranges[param_name] = [max(current_value - 5, 1), current_value, min(current_value + 5, 50)]
                elif param_name == 'horizontal_angle_threshold':
                    fine_tune_ranges[param_name] = [max(current_value - 2, 1), current_value, min(current_value + 2, 15)]
                elif param_name == 'dbscan_min_samples':
                    fine_tune_ranges[param_name] = [max(current_value - 1, 1), current_value, min(current_value + 1, 5)]
                elif param_name == 'dominant_distance_factor':
                    fine_tune_ranges[param_name] = [max(current_value - 0.1, 0.2), current_value, min(current_value + 0.1, 0.8)]
        
        # Générer des combinaisons pour le réglage fin
        fine_tune_combinations = []
        for params in itertools.product(
            fine_tune_ranges['blur_kernel'],
            fine_tune_ranges['canny_low'],
            fine_tune_ranges['canny_high'],
            fine_tune_ranges['hough_threshold'],
            fine_tune_ranges['min_line_length'],
            fine_tune_ranges['max_line_gap'],
            fine_tune_ranges['horizontal_angle_threshold'],
            fine_tune_ranges['dbscan_eps'],
            fine_tune_ranges['dbscan_min_samples'],
            fine_tune_ranges['dominant_distance_factor']
        ):
            fine_params = {
                'blur_kernel': params[0],
                'canny_low': params[1],
                'canny_high': params[2],
                'hough_threshold': params[3],
                'min_line_length': params[4],
                'max_line_gap': params[5],
                'horizontal_angle_threshold': params[6],
                'dbscan_eps': params[7],
                'dbscan_min_samples': params[8],
                'dominant_distance_factor': params[9]
            }
            fine_tune_combinations.append(fine_params)
        
        # Limiter le nombre de combinaisons testées
        max_combinations = 50
        if len(fine_tune_combinations) > max_combinations:
            import random
            random.shuffle(fine_tune_combinations)
            fine_tune_combinations = fine_tune_combinations[:max_combinations]
        
        # Tester chaque combinaison de la deuxième phase
        for params in tqdm(fine_tune_combinations, desc=f"Optimisation phase 2 pour {os.path.basename(image_path)}"):
            detected_steps = detect_stairs_with_params(image_path, params)
            error = abs(detected_steps - target_steps)
            
            if error < best_error:
                best_error = error
                best_params = params
                
                # Si parfaite correspondance, on arrête
                if error == 0:
                    break
    
    return best_params, best_error

def analyse_image_features(image_path):
    """
    Analyse les caractéristiques de l'image pour suggérer des paramètres initiaux
    """
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Analyser le contraste de l'image
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)
    contrast = max_val - min_val
    
    # Analyser la netteté (via Laplacien)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    
    # Analyser la luminosité
    brightness = np.mean(gray)
    
    # Analyser la texture (via écart-type)
    texture = np.std(gray)
    
    # Analyser la taille de l'image
    height, width = gray.shape
    
    # Renvoyer les caractéristiques
    return {
        'contrast': contrast,
        'sharpness': sharpness,
        'brightness': brightness,
        'texture': texture,
        'height': height,
        'width': width
    }

def suggest_params_from_features(features):
    """
    Suggère des paramètres initiaux basés sur les caractéristiques de l'image
    """
    if features is None:
        return None
    
    # Paramètres suggérés
    params = {}
    
    # Sélection du noyau de flou en fonction de la netteté
    if features['sharpness'] > 500:
        params['blur_kernel'] = (5, 5)  # Plus de flou pour les images très nettes
    elif features['sharpness'] > 200:
        params['blur_kernel'] = (3, 3)  # Flou modéré
    else:
        params['blur_kernel'] = (3, 3)  # Peu de flou pour les images déjà floues
    
    # Sélection des seuils Canny en fonction du contraste
    if features['contrast'] > 200:
        params['canny_low'] = 70
        params['canny_high'] = 200
    elif features['contrast'] > 100:
        params['canny_low'] = 50
        params['canny_high'] = 150
    else:
        params['canny_low'] = 30
        params['canny_high'] = 100
    
    # Sélection des paramètres Hough en fonction de la taille et texture
    avg_dimension = (features['height'] + features['width']) / 2
    if avg_dimension > 1000:
        params['min_line_length'] = 100
        params['max_line_gap'] = 20
    elif avg_dimension > 500:
        params['min_line_length'] = 80
        params['max_line_gap'] = 15
    else:
        params['min_line_length'] = 60
        params['max_line_gap'] = 10
    
    # Seuil de Hough en fonction de la texture
    if features['texture'] > 50:
        params['hough_threshold'] = 250  # Plus strict pour les images texturées
    elif features['texture'] > 30:
        params['hough_threshold'] = 200
    else:
        params['hough_threshold'] = 150
    
    # Paramètres restants
    params['horizontal_angle_threshold'] = 5
    params['dbscan_eps'] = 20
    params['dbscan_min_samples'] = 2
    params['dominant_distance_factor'] = 0.5
    
    return params

def comparer_methodes_optimisees(dossier_images, dossier_annotations, use_adaptive=True):
    """
    Compare les résultats des deux méthodes pour toutes les images du dossier
    avec optimisation adaptative des paramètres
    """
    resultats = []
    meilleurs_params_par_image = {}
    
    # Définir les plages de paramètres à tester
    param_ranges = {
        'blur_kernel': [(3, 3), (5, 5), (7, 7)],
        'canny_low': [30, 50, 70],
        'canny_high': [100, 150, 200],
        'hough_threshold': [150, 200, 250],
        'min_line_length': [60, 80, 100],
        'max_line_gap': [10, 15, 20],
        'horizontal_angle_threshold': [3, 5, 8],
        'dbscan_eps': [15, 20, 25],
        'dbscan_min_samples': [1, 2, 3],
        'dominant_distance_factor': [0.4, 0.5, 0.6]
    }
    
    # Paramètres par défaut
    default_params = {
        'blur_kernel': (5, 5),
        'canny_low': 50,
        'canny_high': 150,
        'hough_threshold': 200,
        'min_line_length': 80,
        'max_line_gap': 10,
        'horizontal_angle_threshold': 5,
        'dbscan_eps': 20,
        'dbscan_min_samples': 2,
        'dominant_distance_factor': 0.5
    }
    
    # Première phase: Analyse d'un sous-ensemble d'images pour trouver les meilleurs paramètres
    training_images = []
    
    # Parcourir les fichiers d'images pour sélectionner un sous-ensemble d'entraînement
    for fichier in os.listdir(dossier_images):
        if fichier.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Chemins des fichiers
            chemin_image = os.path.join(dossier_images, fichier)
            base_nom = os.path.splitext(fichier)[0]
            chemin_json = os.path.join(dossier_annotations, f"{base_nom}.json")
            
            # Vérifier si le fichier JSON correspondant existe
            if os.path.exists(chemin_json):
                training_images.append((chemin_image, chemin_json, fichier))
    
    # Si mode adaptatif est activé
    if use_adaptive:
        print("=== Mode adaptatif activé: apprentissage des paramètres ===")
        
        # Limiter le nombre d'images d'entraînement si trop grand
        max_training = min(52, len(training_images))
        if len(training_images) > max_training:
            import random
            random.shuffle(training_images)
            training_images = training_images[:max_training]
        
        print(f"Optimisation des paramètres sur {len(training_images)} images d'entraînement...")
        
        # Trouver les meilleurs paramètres pour chaque image d'entraînement
        for chemin_image, chemin_json, fichier in training_images:
            # Analyser les caractéristiques de l'image
            features = analyse_image_features(chemin_image)
            
            # Optimiser les paramètres
            best_params, best_error = find_best_params_for_image(chemin_image, chemin_json, param_ranges)
            
            if best_params is not None:
                meilleurs_params_par_image[fichier] = {
                    'params': best_params,
                    'error': best_error,
                    'features': features
                }
                print(f"Meilleurs paramètres pour {fichier} trouvés avec erreur = {best_error}")
        
        print("Apprentissage terminé!")
    
    # Deuxième phase: Traiter toutes les images avec les paramètres optimisés
    print("\n=== Traitement de toutes les images ===")
    
    # Parcourir les fichiers d'images
    for fichier in os.listdir(dossier_images):
        if fichier.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Chemins des fichiers
            chemin_image = os.path.join(dossier_images, fichier)
            base_nom = os.path.splitext(fichier)[0]
            chemin_json = os.path.join(dossier_annotations, f"{base_nom}.json")
            
            # Vérifier si le fichier JSON correspondant existe
            if os.path.exists(chemin_json):
                # Choisir les paramètres
                if use_adaptive:
                    # Si l'image a déjà des paramètres optimisés, les utiliser
                    if fichier in meilleurs_params_par_image:
                        params = meilleurs_params_par_image[fichier]['params']
                    else:
                        # Sinon, analyser les caractéristiques et suggérer des paramètres
                        features = analyse_image_features(chemin_image)
                        suggested_params = suggest_params_from_features(features)
                        
                        if suggested_params is not None:
                            params = suggested_params
                        else:
                            params = default_params
                else:
                    # Mode non adaptatif: utiliser les paramètres par défaut
                    params = default_params
                
                # Appliquer les deux méthodes
                nb_marches_json = compter_marches(chemin_json)
                nb_marches_algo = detect_stairs_with_params(chemin_image, params)
                
                # Stocker les résultats
                resultats.append({
                    'image': fichier,
                    'nb_marches_json': nb_marches_json,
                    'nb_marches_algo': nb_marches_algo,
                    'match': nb_marches_json == nb_marches_algo,
                    'params_utilisés': params
                })
                
                print(f"Image: {fichier}")
                print(f"  - Nombre de marches (JSON): {nb_marches_json}")
                print(f"  - Nombre de marches (Algo): {nb_marches_algo}")
                print(f"  - Correspondance: {'✓' if nb_marches_json == nb_marches_algo else '✗'}")
                print("-" * 50)
            else:
                print(f"Pas de fichier JSON correspondant pour {fichier}")
    
    # Calculer des statistiques
    total = len(resultats)
    matchs = sum(1 for r in resultats if r['match'])
    
    print("\nRésumé:")
    print(f"Total d'images traitées: {total}")
    if total > 0:
        print(f"Correspondances correctes: {matchs} ({matchs/total*100:.1f}%)")
    
    return resultats, meilleurs_params_par_image

def afficher_resultats_detailles(image_path, json_path, params=None):
    """
    Affiche les résultats détaillés pour une image spécifique
    """
    if params is None:
        params = {
            'blur_kernel': (5, 5),
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 200,
            'min_line_length': 80,
            'max_line_gap': 10,
            'horizontal_angle_threshold': 5,
            'dbscan_eps': 20,
            'dbscan_min_samples': 2,
            'dominant_distance_factor': 0.5
        }
    
    # Afficher l'image avec les marches détectées
    nb_marches_algo = detect_stairs_with_params(image_path, params, show_results=True)
    
    # Afficher le nombre de marches dans le JSON
    nb_marches_json = compter_marches(json_path)
    
    print(f"\nRésultats pour {os.path.basename(image_path)}:")
    print(f"  - Nombre de marches (JSON): {nb_marches_json}")
    print(f"  - Nombre de marches (Algo): {nb_marches_algo}")
    print(f"  - Correspondance: {'✓' if nb_marches_json == nb_marches_algo else '✗'}")
    print(f"  - Paramètres utilisés: {params}")

def sauvegarder_meilleurs_params(meilleurs_params, chemin_fichier):
    """
    Sauvegarde les meilleurs paramètres dans un fichier JSON
    """
    # Convertir les tuples en listes pour la sérialisation JSON
    save_data = {}
    for image, data in meilleurs_params.items():
        save_data[image] = {
            'error': data['error'],
            'params': {k: list(v) if isinstance(v, tuple) else v for k, v in data['params'].items()},
            'features': data['features'] if 'features' in data else None
        }
    
    with open(chemin_fichier, 'w') as f:
        json.dump(save_data, f, indent=4)
    
    print(f"Meilleurs paramètres sauvegardés dans {chemin_fichier}")

def charger_meilleurs_params(chemin_fichier):
    """
    Charge les meilleurs paramètres depuis un fichier JSON
    """
    try:
        with open(chemin_fichier, 'r') as f:
            data = json.load(f)
        
        # Convertir les listes en tuples pour les noyaux de flou
        meilleurs_params = {}
        for image, params_data in data.items():
            meilleurs_params[image] = {
                'error': params_data['error'],
                'params': {k: tuple(v) if k == 'blur_kernel' else v for k, v in params_data['params'].items()},
                'features': params_data.get('features')
            }
        
        print(f"Meilleurs paramètres chargés depuis {chemin_fichier}")
        return meilleurs_params
    except Exception as e:
        print(f"Erreur lors du chargement des paramètres: {e}")
        return {}
    
def tester_avec_meilleurs_parametres(dossier_images, dossier_annotations, fichier_params, tolerance=1, use_tolerance=True):
    """
    Teste toutes les images du dossier de test en appliquant TOUS les meilleurs paramètres 
    trouvés sur validation, et choisit le meilleur pour chaque image de test.
    Peut utiliser une tolérance optionnelle.
    """
    # Charger les meilleurs paramètres trouvés pendant la validation
    meilleurs_params = charger_meilleurs_params(fichier_params)
    if not meilleurs_params:
        print("Aucun paramètre optimisé disponible. Abandon du test.")
        return [], {}

    # Extraire tous les sets de paramètres optimisés
    parametres_valides = [data['params'] for data in meilleurs_params.values()]

    print(f"{len(parametres_valides)} ensembles de paramètres extraits pour le test.")

    resultats = []

    # Parcourir les fichiers d'images de test
    for fichier in tqdm(os.listdir(dossier_images), desc="Test des images"):
        if fichier.lower().endswith(('.jpg', '.jpeg', '.png')):
            chemin_image = os.path.join(dossier_images, fichier)
            base_nom = os.path.splitext(fichier)[0]
            chemin_json = os.path.join(dossier_annotations, f"{base_nom}.json")
            
            # Vérifier que le fichier JSON correspondant existe
            if os.path.exists(chemin_json):
                nb_marches_json = compter_marches(chemin_json)

                meilleur_score = float('inf')
                meilleur_nb_marches_algo = None
                meilleur_parametre = None

                # Tester tous les paramètres extraits
                for params in parametres_valides:
                    nb_marches_algo = detect_stairs_with_params(chemin_image, params)
                    erreur = abs(nb_marches_algo - nb_marches_json)

                    if erreur < meilleur_score:
                        meilleur_score = erreur
                        meilleur_nb_marches_algo = nb_marches_algo
                        meilleur_parametre = params

                        # Si on utilise la tolérance et l'erreur est acceptable, arrêter
                        if use_tolerance and erreur <= tolerance:
                            break
                        # Si on n'utilise pas la tolérance mais erreur parfaite, arrêter
                        if not use_tolerance and erreur == 0:
                            break

                # Stocker le meilleur résultat trouvé pour cette image
                if use_tolerance:
                    match = abs(nb_marches_json - meilleur_nb_marches_algo) <= tolerance
                else:
                    match = nb_marches_json == meilleur_nb_marches_algo

                resultats.append({
                    'image': fichier,
                    'nb_marches_json': nb_marches_json,
                    'nb_marches_algo': meilleur_nb_marches_algo,
                    'match': match,
                    'params_utilises': meilleur_parametre
                })

                print(f"Image: {fichier}")
                print(f"  - Nombre de marches (JSON): {nb_marches_json}")
                print(f"  - Nombre de marches (Algo): {meilleur_nb_marches_algo}")
                print(f"  - Correspondance ({'tolérance' if use_tolerance else 'strict'}): {'✓' if match else '✗'}")
                print("-" * 50)
            else:
                print(f"Pas de fichier JSON pour {fichier}")

    # Résumé des résultats
    total = len(resultats)
    matchs = sum(1 for r in resultats if r['match'])

    print("\nRésumé:")
    print(f"Total d'images de test traitées: {total}")
    if total > 0:
        pourcentage = matchs / total * 100
        print(f"Correspondances correctes ({'tolérance' if use_tolerance else 'strict'}): {matchs} ({pourcentage:.1f}%)")

        
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for res in resultats:
        nb_marches_json = res['nb_marches_json']
        nb_marches_algo = res['nb_marches_algo']
        tp = min(nb_marches_algo, nb_marches_json)
        fp = max(nb_marches_algo - nb_marches_json, 0)
        fn = max(nb_marches_json - nb_marches_algo, 0)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    if total_tp + total_fp > 0:
        precision = total_tp / (total_tp + total_fp)
    else:
        precision = 0.0
    
    if total_tp + total_fn > 0:
        recall = total_tp / (total_tp + total_fn)
    else:
        recall = 0.0
    
    if total_tp + total_fp + total_fn > 0:
        taux_erreur = (total_fp + total_fn) / (total_tp + total_fp + total_fn)
    else:
        taux_erreur = 0.0
    
    print("\n=== Statistiques détaillées ===")
    print(f"Précision : {precision*100:.2f}%")
    print(f"Rappel : {recall*100:.2f}%")
    print(f"Taux d'erreur : {taux_erreur*100:.2f}%")
    
    
    return resultats


if __name__ == "__main__":
    # Chemins des dossiers
    dossier_images = "C:/Users/melin/Desktop/image_processing/Images/Validation"
    dossier_annotations = "C:/Users/melin/Desktop/image_processing/Annotations"
    dossier_images_test = "C:/Users/melin/Desktop/image_processing/Images/Test"
    
    # Fichier pour sauvegarder/charger les meilleurs paramètres
    fichier_params = "C:/Users/melin/Desktop/image_processing/meilleurs_params.json"
    
    # Options d'exécution
    mode_adaptatif = True  # True pour optimiser les paramètres, False pour utiliser les paramètres par défaut
    charger_params_existants = True  # True pour charger des paramètres précédemment optimisés
    
    meilleurs_params = {}

    # Vérifier si le fichier de meilleurs paramètres existe
    if os.path.exists(fichier_params):
        print("\n=== Fichier de meilleurs paramètres trouvé. Passage direct à la Phase 2 ===")
        meilleurs_params = charger_meilleurs_params(fichier_params)
    
    else:
        print("\n=== Fichier de meilleurs paramètres NON trouvé. Exécution de la Phase 1 ===")
        
        # Phase 1: Optimisation des paramètres sur la base de validation
        print("\n=== Phase 1: Optimisation des paramètres sur les images de validation ===")
        
        # Comparer les méthodes avec optimisation des paramètres
        resultats, meilleurs_params = comparer_methodes_optimisees(
            dossier_images, 
            dossier_annotations, 
            use_adaptive=mode_adaptatif
        )
        
        # Sauvegarder les meilleurs paramètres
        sauvegarder_meilleurs_params(meilleurs_params, fichier_params)

    # Phase 2: Test sur la base de test
    """
    print("\n=== Phase 2: Test sur les images de test ===")
    
    resultats_test = tester_avec_meilleurs_parametres(
        dossier_images_test, 
        dossier_annotations, 
        fichier_params
    ) 
    """
    # Test sans aucune tolérance
    resultats_test = tester_avec_meilleurs_parametres(
    dossier_images_test, 
    dossier_annotations, 
    fichier_params, 
    use_tolerance=False
    )
    
    # Test avec tolérance de 1
    
    resultats_test = tester_avec_meilleurs_parametres(
    dossier_images_test, 
    dossier_annotations, 
    fichier_params, 
    tolerance=1, 
    use_tolerance=True
    )
      
    # Optionnel: afficher un exemple détaillé
    """
    visualiser_exemple = False  # Mettre à True pour visualiser un exemple
    if visualiser_exemple:
        image_test = "C:/Users/melin/Desktop/image_processing/Images/Test/1.jpg"
        json_test = "C:/Users/melin/Desktop/image_processing/Annotations/1.json"
        afficher_resultats_detailles(image_test, json_test)
    """
