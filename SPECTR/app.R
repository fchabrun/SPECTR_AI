# All rights reserved by Floris Chabrun <floris.chabrun@chu-angers.fr>
#
# Author : Floris Chabrun <floris.chabrun@chu-angers.fr>, Xavier Dieu <xavier.dieu@chu-angers.fr>
# CHU Angers, University Hospital, Angers, France

##################################################################################################
######################################## CONSTANTS ###############################################
##################################################################################################

# sink("C:/users/admin/downloads/temp.txt")

# RUN
USE_SHINYJS=T
VERBOSE = 2
QUICK_MINIATURES=T
POPUPS_TRIGGER = c("focus","hover")[2]
POPUPS_OPTIONS = list(container="body")
POPUPS_DELAY_IN = 500
POPUPS_DELAY_OUT = 0

#### FOR CHOOSING BETWEEN ONLINE/OFFLINE
LOCAL=(Sys.getenv('SHINY_PORT') == "") # set this to F for uploading online
# below should not change, even for local use
DEBUG_MODE=F # set this to F for uploading online
RETICULATE=T # set this to T for uploading online
TWOSTEP_ANALYSIS=T # first load models, then every time we upload a batch use the same models (do not reload them)
#### FOR CHOOSING BETWEEN ONLINE/OFFLINE

DELAY_BETWEEN_INPUTS = 0 # in milliseconds
DELAY_AFTER_REFRESH = 1000 # in milliseconds

LICENSE_TEXT = paste0("<br><br>",
                      "This software named SPECTR version 0.91 dated July 05th, 2019 have been developed by CHU d&rsquo;Angers and Universit&eacute; d&rsquo;Angers. ",
                      "The copyright and all exploitation/use rights of this software are and remain their property. It has been registered the 24th of October, 2019 at the French protection program agency under the number IDDN.FR.001.440032.000.S.P.2019.000.31230.",
                      "<br><br>",
                      "The access to the software is granted only to be used for <u>internal academic research purposes</u> and access to source code is not granted.",
                      "<br><br>",
                      "In particular, any of the following uses of the software <u>are prohibited</u>:",
                      "<br><br>",
                      "<ul>",
                      "<li><u>commercial purposes</u>;</li>",
                      "<li><u>use in a clinical context</u>, meaning notably it is prohibited to use the software for patient treatment or diagnosis;</li>",
                      "<li>translation, adaptation, arrangement, modification, correction of errors and bugs;</li>",
                      "<li>grant software or derived product sub-sub-licenses to any third party or affiliate;</li>",
                      "<li>use the software in partnership project(s);</li>",
                      "<li>proposing any service concerning the software to any third party or affiliate.</li>",
                      "</ul>",
                      "<br><br><br>")

printd <- function(text, level=2) {
  if (level <= VERBOSE) {
    print(text)
  }
}

# DEMO BATCH location
expert_batch_path = "data/hem-batch-internal.json"
# expert_batch_path = "data/expert-batch-internal-001.json"
expert_batch_path2 = "data/expert-batch-internal-001b.json"
demo_batch_path = "data/test-batch-internal-002.json"
demo_json_template_path = "data/test-batch-internal-002.json"
demo_img_template_path = "data/template.png"
demo_img_tutorial = "data/SPECTR_ImageCroppingTutorial.pdf"

# FOR COMMUNICATING WITH PYTHON
if (!RETICULATE) {
  python_path = "C:/Users/admin/anaconda3/envs/py37_tf23_feb21/python.exe"
}
py_script_path = "ai.py"
py_models_path = "models"
py_temp_path = "temp"

thumbnail_template_path <- "data/blank_sample.jpg"

# SOFTWARE
MAIN_VERSION = "0"
SUB_VERSION = "10"
MINOR_VERSION = "1"

LOCALIZATION = 'EN'

ai <- NULL
loaded_models <- NULL
# if (RETICULATE & TWOSTEP_ANALYSIS) {
#   # ai models
# }

# Misc
SPE_WIDTH = 304

available_classes_input <- c('Normal', 'Heterogeneity restriction aspect', 'Clonal anomaly', 'Beta-gamma bridge')
available_classes_input_short <- c('Normal', 'Heterogeneity', 'Clonal', 'Beta-gamma')

predmap_classes=c('None', 'Fractions', 'Spikes')

# REFERENCE VALUES
adult_spe_standards_pct = data.frame(std_lo = c(55.8, 2.9, 7.1, 4.7, 3.2, 11.1),
                                     std_hi = c(66.1, 4.9, 11.8, 7.2, 6.5, 18.8))
adult_spe_standards_abs = data.frame(std_lo = c(40.2, 2.1, 5.1, 3.4, 2.3, 8.0),
                                     std_hi = c(47.6, 3.5, 8.5, 5.2, 4.7, 13.5))

age.0m.6m_spe_standards_pct = data.frame(std_lo = c(58.9, 3.2, 10.6, 4.8, 2.1, 3.5),
                                         std_hi = c(73.4, 11.7, 14.0, 7.9, 3.3, 9.7))
age.0m.6m_spe_standards_abs = data.frame(std_lo = c(27.3, 2.1, 5.3, 2.2, 1.1, 1.7),
                                         std_hi = c(49.1, 5.4, 9.8, 4.6, 2.1, 6.3))

age.6m.1y_spe_standards_pct = data.frame(std_lo = c(57.4, 3.0, 10.2, 5.3, 2.1, 4.2),
                                         std_hi = c(71.4, 5.0, 16.1, 6.9, 3.6, 11.0))
age.6m.1y_spe_standards_abs = data.frame(std_lo = c(36.0, 2.0, 6.3, 3.3, 1.4, 2.8),
                                         std_hi = c(50.6, 3.7, 12.1, 4.9, 2.6, 8.0))

age.1y.2y_spe_standards_pct = data.frame(std_lo = c(57.4, 3.2, 10.7, 5.6, 2.3, 5.8),
                                         std_hi = c(69.0, 5.4, 15.5, 7.0, 3.5, 12.1))
age.1y.2y_spe_standards_abs = data.frame(std_lo = c(38.7, 2.4, 7.8, 3.7, 1.6, 4.2),
                                         std_hi = c(51.5, 4.0, 11.6, 5.2, 2.7, 8.8))

age.2y.7y_spe_standards_pct = data.frame(std_lo = c(57.5, 3.3, 10.0, 5.2, 2.6, 7.7),
                                         std_hi = c(67.7, 5.4, 14.8, 7.0, 4.2, 14.8))
age.2y.7y_spe_standards_abs = data.frame(std_lo = c(30.5, 2.0, 5.6, 2.8, 1.5, 4.6),
                                         std_hi = c(48.9, 3.7, 10.6, 5.2, 3.1, 10.7))

age.7y.15y_spe_standards_pct = data.frame(std_lo = c(57.1, 3.2, 8.9, 5.1, 2.9, 9.8),
                                          std_hi = c(67.2, 4.9, 13.0, 6.9, 5.2, 16.9))
age.7y.15y_spe_standards_abs = data.frame(std_lo = c(30.9, 1.7, 4.8, 2.7, 1.7, 6.0),
                                          std_hi = c(49.5, 3.7, 9.7, 5.2, 3.9, 12.7))

ref_values <- function(age, abs) {
  if (age<.5) {
    if (abs) {
      return(age.0m.6m_spe_standards_abs)
    } else {
      return(age.0m.6m_spe_standards_pct)
    }
  } else if (age<1) {
    if (abs) {
      return(age.6m.1y_spe_standards_abs)
    } else {
      return(age.6m.1y_spe_standards_pct)
    }
  } else if (age<2) {
    if (abs) {
      return(age.1y.2y_spe_standards_abs)
    } else {
      return(age.1y.2y_spe_standards_pct)
    }
  } else if (age<7) {
    if (abs) {
      return(age.2y.7y_spe_standards_abs)
    } else {
      return(age.2y.7y_spe_standards_pct)
    }
  } else if (age<18) {
    if (abs) {
      return(age.7y.15y_spe_standards_abs)
    } else {
      return(age.7y.15y_spe_standards_pct)
    }
  } else {
    if (abs) {
      return(adult_spe_standards_abs)
    } else {
      return(adult_spe_standards_pct)
    }
  }
}

ref_values_text <- function(age, abs) {
  tmp_ref_df <- ref_values(age, abs)
  return(paste0(tmp_ref_df$std_lo, '-', tmp_ref_df$std_hi))
}

##################################################################################################
########################################### MAIN #################################################
##################################################################################################

printd('Loading required libraries...')

if (RETICULATE) {
  library(reticulate)
  library(tensorflow)
  library(keras)
  
  if (!LOCAL) {
    # Create a virtual environment selecting your desired python version
    reticulate::virtualenv_create(envname = "python_environment", python = 'python3')
    
    # try to solve h5py problem
    # reticulate::virtualenv_remove(packages="h5py", envname = "python_environment")
    # reticulate::virtualenv_install("h5py", version = "2.1.0", envname = "python_environment")
    
    # Explicitly install python libraries that you want to use, e.g. pandas, numpy
    # virtualenv_install("python_environment", packages = c('numpy','tensorflow'))
    # Select the virtual environment
    reticulate::use_virtualenv("python_environment", required = TRUE)
  } else {
    library(reticulate)
    use_python("C:/Users/admin/anaconda3/envs/py37_tf23_feb21/python.exe")
    # use_condaenv("py37_tf23_feb21")
  }
}

library(shiny)
if (USE_SHINYJS) {
  library(shinyjs)
}
library(shinyBS)
library(RJSONIO)
library(ggplot2)
library(DT)
library(magick)

printd(paste0('Running with ShinyJS: ', USE_SHINYJS))

##################################################################################################
######################################## AI BACKEND ##############################################
##################################################################################################

# CONSTANTS
MAX_CONFIDENCE_THRESHOLD=.1
MAX_CONFIDENCE_UNCERTAINTY=.005
UNCERTAINTY_COUNT_THRESHOLD=5
H_THRESHOLD=.5
S_THRESHOLD=.5

R_quantifyFractions <- function(x, y) {
  area <- (x[-length(x)]+x[-1])/2
  total_area = sum(area)
  fractions_area = numeric(length(y)+1)
  fractions_points = c(1, y, length(x))
  for(i in 1:(length(fractions_points)-1)) {
    fractions_area[i] = sum(area[fractions_points[i]:(fractions_points[i+1]-1)])
  }
  fractions_percent = fractions_area / total_area
  return(fractions_percent)
}

R_adjustSegmentationPoints <- function(x, y, adjust_max = 10) {
  # compute first derivative
  d = diff(x)
  # detect valleys
  # look for indices for which d[i-1] < 0 & d[i] >= 0 (valleys)
  valleys <- which(d[-length(d)] < 0 & d[-1] >= 0) + 1
  # create new y, by default initialized to -1
  new_y = rep(-1, length(y))
  # now that the valleys have been determined, we will tr to match each predicted y point to those valleys
  # i.e. adjust each y to the closest valley
  valleys_matrix <- matrix(rep(valleys, each = length(y)), length(y))
  valleys_matrix <- abs(valleys_matrix-y)
  # match
  while(T) {
    valleys_min = min(valleys_matrix)
    if (valleys_min > adjust_max) {
      # out of range
      break
    }
    next_index = which(valleys_matrix == valleys_min, arr.ind = T)[1,]
    i = next_index[1]
    j = next_index[2]
    new_y[i] = valleys[j]
    valleys_matrix[i,] = adjust_max + 1
    valleys_matrix[,j] = adjust_max + 1
  }
  flag <- 0
  # raise warning flag if no valley detected close to a point
  if (any(new_y == -1))
    flag <- 1
  # if there were unadjusted values, recall initial values
  new_y <- ifelse(new_y == -1, y, new_y)
  return(list(new_y = new_y, flag = flag))
}

R_segmentation_maps_to_fraction_boundaries <- function(y_) {
  spe_width=304
  y_map_fractions=array(0, c(length(y_), spe_width))
  y_map_confidence=array(0, c(length(y_), spe_width))
  y_boundaries=array(0, c(length(y_), 5))
  y_uncertainty=numeric(length(y_))
  y_confidence=numeric(length(y_))
  
  for(ix in 1:length(y_)) {
    y_map=max.col(y_[[ix]])
    # if confidence < a certain threshold, display "unknown fraction"
    y_map_conf=apply(y_[[ix]], 1, max)
    y_map[y_map_conf<MAX_CONFIDENCE_THRESHOLD]=0
    # save
    y_map_fractions[ix,]=y_map
    y_map_confidence[ix,]=y_map_conf
    # then list all possible limits between fractions, i.e. as soon as there is a variation
    interfaces=which(diff(y_map)!=0)
    # compute the number of possibilities
    poss=0
    for(b1 in 1:(length(interfaces)-4))
      for(b2 in (b1+1):(length(interfaces)-3))
        for(b3 in (b2+1):(length(interfaces)-2))
          for(b4 in (b3+1):(length(interfaces)-1))
            for(b5 in (b4+1):(length(interfaces)))
              poss=poss+1
    # then list all possibilities, and for each, compute the percentage of points correctly placed
    C=numeric(poss)
    B=array(0, dim=c(poss,5))
    poss=1
    for(b1 in 1:(length(interfaces)-4)) {
      for(b2 in (b1+1):(length(interfaces)-3)) {
        for(b3 in (b2+1):(length(interfaces)-2)) {
          for(b4 in (b3+1):(length(interfaces)-1)) {
            for(b5 in (b4+1):(length(interfaces))) {
              b=c(1, interfaces[b1], interfaces[b2], interfaces[b3], interfaces[b4], interfaces[b5], spe_width)
              n=numeric(6)
              for(i in 1:6)
                n[i]=sum(y_map[b[i]:b[i+1]]==i)
              C[poss]=(sum(n)+sum(y_map==0))/spe_width
              B[poss, ]=b[2:6]
              poss=poss+1
            }
          }
        }
      }
    }
    # compute fractions
    if (sum(abs(C-max(C))<MAX_CONFIDENCE_UNCERTAINTY)>1 & sum(y_map==0)>UNCERTAINTY_COUNT_THRESHOLD) {
      y_uncertainty[ix]=1
    }
    #printd('found uncertainty for curve: '+str(ix))
    y_confidence[ix]=max(C)
    y_boundaries[ix, ]=B[which.max(C),]
  }
  return(list(y_boundaries, y_uncertainty, y_confidence, y_map_fractions, y_map_confidence))
}

R_analyze_raw_predictions <- function(cnn_results, batch_metadata, auto_adjust=F) {
  RESULTS = list()
  # Get size of batch
  n=length(cnn_results)
  spe_width = 304
  # SEGMENTATION
  # Predict
  raw_f_predictions <- lapply(1:n, function(ix) {as.matrix(do.call(cbind, list(cnn_results[[ix]]$f1,cnn_results[[ix]]$f2,cnn_results[[ix]]$f3,cnn_results[[ix]]$f4,cnn_results[[ix]]$f5,cnn_results[[ix]]$f6)))})
  # Convert output maps -> fraction points
  f_predictions <- R_segmentation_maps_to_fraction_boundaries(raw_f_predictions)
  # Adjust and check, compute concentrations
  if (auto_adjust) {
    for(ix in 1:n) {
      tmp=R_adjustSegmentationPoints(cnn_results[[ix]]$x, f_predictions[[1]][ix,])
      f_predictions[[1]][ix,]=tmp$new_y
    }
  }
  
  # CLASSIFICATION
  # Predict
  c_predictions <- do.call(rbind, lapply(1:n, function(ix) {cnn_results[[ix]]$c}))
  
  # HEMOLYSIS
  # Predict
  # h_predictions <- do.call(rbind, lapply(1:n, function(ix) {cnn_results[[ix]]$h}))
  h_raw_predictions <- sapply(1:n, function(ix) {cnn_results[[ix]]$h})
  h_predictions=cbind(h_raw_predictions, (h_raw_predictions>H_THRESHOLD)*1)
  
  # SPIKES
  # Predict
  raw_s_predictions <- do.call(rbind, lapply(1:n, function(ix) {cnn_results[[ix]]$s}))
  # Read maps
  MAX_PEAKS = 32
  s_predictions=array(0, dim=c(n,MAX_PEAKS*3))
  for(ix in 1:n) {
    # compute the binary map (0/1)
    s_map=raw_s_predictions[ix,]
    # then analyze first derivative to find start/ends of peaks
    s_diff=diff(1*(s_map>S_THRESHOLD))
    if (any(s_diff!=0)) {
      # determine beginning and end of each peak
      s_start=which(s_diff==1)+1
      s_end=which(s_diff==-1)
      # compute mean confidence of each peak
      for(i in 1:length(s_start)) {
        if (i>MAX_PEAKS)
          break
        s_predictions[ix,3*(i-1)+1]=mean(s_map[s_start[i]:s_end[i]]) # confiance
        s_predictions[ix,3*(i-1)+2]=s_start[i]+1
        s_predictions[ix,3*(i-1)+3]=s_end[i]+1
      }
    }
  }
  
  return(list(f_predictions[[1]], c_predictions, s_predictions, h_predictions, f_predictions[[2]], f_predictions[[3]], f_predictions[[4]], f_predictions[[5]], raw_s_predictions))
}


##################################################################################################
######################################## PDF BACKEND #############################################
##################################################################################################

IMG2SPECTR <- function(raw_buffer) {
  require(magick)
  # img_path = "C:/Users/admin/Downloads/pic_test_18.png"
  # raw_buffer=image_read(img_path)
  # convert to black and white
  proc_buffer <- raw_buffer %>%
    # image_scale("300") %>% # resized to width=300 (300 values)
    image_quantize(colorspace = 'gray') # convert to grayscale
  # convert to matrix
  mat = image_data(proc_buffer, 'gray')[1,,]
  mat = matrix(as.numeric(mat), ncol=ncol(mat),nrow=nrow(mat))
  # convert to 1 and invert (white>->black)
  mat = (255-mat)/255
  # remove "base line"
  new_baseline = ncol(mat)
  for(i in ncol(mat):1) {
    if (any(mat[ ,i]==0)) { # if all values of the row are >0 -> baseline, remove values
      break
    }
    new_baseline = i-1
    mat[ ,i] <- 0
  }
  # remove "vertical lines" (i.e. fractions) if present
  curve_base_thickness = median(rowSums(mat>0))
  for(pos in 1:nrow(mat)) {
    cons=T
    for(j in 1:5) {
      if (mat[pos,new_baseline-j]==0) {
        cons=F
        break
      }
    }
    if(cons) {
      n_kept=0
      for(j in 1:ncol(mat)) {
        if (mat[pos,j]>0) {
          if (n_kept<curve_base_thickness) {
            n_kept<-n_kept+1
          } else {
            mat[pos,j] = 0
          }
        }
      }
    }
  }
  # analyze
  values = numeric(0)
  for(pos in 1:nrow(mat)) {
    value = 0
    non_null_values <- which(mat[pos,]>0)
    if (length(non_null_values) > 0) {
      # keep only max values (i.e. darker)
      # non_null_values = non_null_values[mat[pos,non_null_values] == max(mat[pos,non_null_values])]
      raw_value <- median(non_null_values)
      value <- ncol(mat)-raw_value
    }
    values[length(values)+1] <- value
  }
  # reverse
  values <- rev(values)
  # rescale
  new_res <- 300
  final_curve <- rep(0, new_res)
  for(i in 1:new_res) {
    new_i <- i * length(values)/new_res
    if (new_i%%1==0) {
      final_curve[i] <- values[new_i]
    } else {
      final_curve[i] <- (1-new_i%%1) * values[floor(new_i)] + new_i%%1 * values[floor(new_i)+1]
    }
  }
  # normalize
  final_curve <- final_curve/max(final_curve)
  if (F) {
    values[500:520]
    ggplot() +
      geom_line(data=data.frame(x=1:length(values), y=values), aes(x=x,y=y)) +
      theme_minimal()
    ggplot() +
      geom_line(data=data.frame(x=1:length(final_curve), y=final_curve), aes(x=x,y=y)) +
      theme_minimal()
  }
  final_curve
}

##################################################################################################
####################################### POST BACKEND #############################################
##################################################################################################

# CONSTANTS
printd('post_backend/source')

# Back end functions
r_quantifyFractions <- function(x, y) {
  printd('post_backend/r_quantifyFractions:in')
  
  # printd('x is:', level=1)
  # printd(x, level=1)
  # printd('y is:', level=1)
  # printd(y, level=1)
  
  area <- (x[-length(x)]+x[-1])/2
  total_area = sum(area)
  fractions_area = numeric(length(y)+1)
  fractions_points = c(1, y, length(x))
  for(i in 1:(length(fractions_points)-1)) {
    fractions_area[i] = sum(area[fractions_points[i]:(fractions_points[i+1]-1)])
  }
  fractions_percent = fractions_area / total_area
  
  printd('post_backend/r_quantifyFractions:out')
  
  return(fractions_percent)
}

quantifySpikes <- function(x, s_start, s_end) {
  printd('post_backend/quantifySpikes:in')
  
  area <- (x[-length(x)]+x[-1])/2
  total_area = sum(area)
  
  spikes <- mapply(function(a,b) {
    sum(area[a:(b-1)])
  }, s_start, s_end)
  spikes_pct <- spikes / total_area
  
  printd('post_backend/quantifySpikes:out')
  
  return(spikes_pct)
}

seekBisalbuminemia <- function(x, y, threshold = .0125) {
  printd('post_backend/seekBisalbuminemia:in')
  
  d1 = diff(x)
  d1_A = d1[1:y[1]]
  increase_threashold = threshold
  decrease_threshold = -threshold
  in_peak_increase = FALSE
  in_peak_decrease = FALSE
  flag = 0
  for(i in 1:length(d1_A)) {
    if (d1_A[i] > increase_threashold) {
      # if increasing (i.e. in the first half of a peak)
      if (!in_peak_increase) {
        # we weren't in a peak before ; are we in the first peak (i.e. albumin), or in a second peak (i.e. >1 peak in albumin -> bisalbumniemia?)
        if (in_peak_decrease) {
          # we were decreasing, and we're increasing again : bisalbuminemia?
          flag <- 1
          break
        } else {
          # before first peak: ok
          in_peak_increase <- TRUE
        }
      }
    } else if (d1_A[i] < decrease_threshold) {
      # we're in a peak, decreasing
      in_peak_increase <- FALSE
      in_peak_decrease <- TRUE
    }
  }
  
  printd('post_backend/seekBisalbuminemia:out')
  
  return(flag)
}

checkSegmentation <- function(x, y, adjust_max = 10) {
  printd('post_backend/checkSegmentation:in')
  
  # compute first derivative
  d = diff(x)
  # detect valleys
  # look for indices for which d[i-1] < 0 & d[i] >= 0 (valleys)
  valleys <- which(d[-length(d)] < 0 & d[-1] >= 0) + 1
  valleys_matrix <- matrix(rep(valleys, each = length(y)), length(y))
  valleys_matrix <- abs(valleys_matrix-y)
  
  printd('post_backend/checkSegmentation:out')
  
  return(any(apply(valleys_matrix, 1, min) > adjust_max))
}

sampleAnalysis <- function(tmp, cursor_status, spike_start, batch_index, usedRecommendations, DEBUG_MODE=FALSE, force_plots=FALSE) {
  printd('post_backend/sampleAnalysis:in')
  
  # printd(paste0('analysis called for sample: ', tmp$sample_id))
  # checked cached data
  if (is.null(tmp$cache))
    tmp$cache=list()
  
  cached_boundaries_differ=F
  cached_spikes_differ=F
  cached_class_differ=F
  cached_hemolysis_differ=F
  cached_locked_differ=F
  cached_cursor_status_differ=F
  
  # did boundaries change?
  if (is.null(tmp$cache$boundaries)) {
    cached_boundaries_differ=T
    tmp$cache$boundaries=tmp$boundaries # reset cache
  } else {
    old_f=tmp$boundaries
    new_f=tmp$cache$boundaries
    if (length(old_f)!=length(new_f)) {
      cached_boundaries_differ=T
      tmp$cache$boundaries=tmp$boundaries # reset cache
    } else if (any(old_f!=new_f)) {
      cached_boundaries_differ=T
      tmp$cache$boundaries=tmp$boundaries # reset cache
    }
  }
  
  # did spikes change?
  if (is.null(tmp$cache$spikes)) {
    cached_spikes_differ=T
    tmp$cache$spikes=tmp$spikes # reset cache
  } else {
    old_s=tmp$spikes
    new_s=tmp$cache$spikes
    if (nrow(old_s)!=nrow(new_s)) {
      cached_boundaries_differ=T
      tmp$cache$spikes=tmp$spikes # reset cache
    } else if (any(old_s$start!=new_s$start) | any(old_s$end!=new_s$end)) {
      cached_boundaries_differ=T
      tmp$cache$spikes=tmp$spikes # reset cache
    }
  }
  
  # did class change?
  if (is.null(tmp$cache$class)) {
    cached_class_differ=T
    tmp$cache$class=tmp$class # reset cache
  } else if (tmp$cache$class!=tmp$class) {
    cached_class_differ=T
    tmp$cache$class=tmp$class # reset cache
  }
  
  # did hemolysis change?
  if (is.null(tmp$cache$haemolysis)) {
    cached_hemolysis_differ=T
    tmp$cache$haemolysis=tmp$haemolysis # reset cache
  } else if (tmp$cache$haemolysis!=tmp$haemolysis) {
    cached_hemolysis_differ=T
    tmp$cache$haemolysis=tmp$haemolysis # reset cache
  }
  
  # did locked change?
  if (is.null(tmp$cache$locked)) {
    cached_locked_differ=T
    tmp$cache$locked=tmp$locked # reset cache
  } else if (tmp$cache$locked!=tmp$locked) {
    cached_locked_differ=T
    tmp$cache$locked=tmp$locked # reset cache
  }
  
  # did cursor status change?
  if (is.null(tmp$cache$cursor_status)) {
    cached_cursor_status_differ=T
    tmp$cache$cursor_status=cursor_status # reset cache
  } else if (tmp$cache$cursor_status!=cursor_status) {
    cached_cursor_status_differ=T
    tmp$cache$cursor_status=cursor_status # reset cache
  }
  
  if (tmp$locked==0) {
    # fractions
    if (cached_boundaries_differ) {
      # fraction boundaries have changed
      # extract fractions
      f <- tmp$boundaries
      # recompute percentages
      new_pct <- r_quantifyFractions(tmp$original_curve_y, f)
      # recompute absolute values
      new_abs <- new_pct * tmp$total_protein
      
      # round
      new_pct=round(new_pct,3)
      new_abs=round(new_abs,1)
      
      # update number of fractions and names according to the number of fractions
      n <- length(f)+1
      if (n == 6) {
        if (LOCALIZATION=='FR') {
          new_longnames <- c("Albumine","Alpha-1-globulines","Alpha-2-globulines","Bêta-1-globulines","Bêta-2-globulines","Gammaglobulines")
          new_names <- c("Albumin","Alpha-1","Alpha-2","Beta-1","Beta-2","Gamma")
        } else if (LOCALIZATION=='EN') {
          new_longnames <- c("Albumin","Alpha-1-globulins","Alpha-2-globulins","Beta-1-globulins","Beta-2-globulins","Gammaglobulins")
          new_names <- c("Albumin","Alpha-1","Alpha-2","Beta-1","Beta-2","Gamma")
        }
      } else if (n == 5) {
        if (LOCALIZATION=='FR') {
          new_longnames <- c("Albumine","Alpha-1-globulines","Alpha-2-globulines","Bêta-1-globulines","Bêta-gammaglobulines")
          new_names <- c("Albumin","Alpha-1","Alpha-2","Beta-1","Beta-gamma")
        } else if (LOCALIZATION=='EN') {
          new_longnames <- c("Albumin","Alpha-1-globulins","Alpha-2-globulins","Beta-1-globulins","Beta-gammaglobulins")
          new_names <- c("Albumin","Alpha-1","Alpha-2","Beta-1","Beta-gamma")
        }
      } else if (n == 4) {
        if (LOCALIZATION=='FR') {
          new_longnames <- c("Albumine","Alpha-1-globulines","Alpha-2-globulines","Bêta-gammaglobulines")
          new_names <- c("Albumin","Alpha-1","Alpha-2","Beta-gamma")
        } else if (LOCALIZATION=='EN') {
          new_longnames <- c("Albumin","Alpha-1-globulins","Alpha-2-globulins","Beta-gammaglobulins")
          new_names <- c("Albumin","Alpha-1","Alpha-2","Beta-gamma")
        }
      } else {
        new_names <- rep("?", n)
        new_longnames <- new_names
      }
      
      # store
      tmp$fractions_qty_pct <- new_pct
      tmp$fractions_qty_abs <- new_abs
      tmp$fractions_names <- new_names
      tmp$fractions_shortnames <- new_names
      tmp$fractions_longnames <- new_longnames
    }
    
    # spikes
    if (cached_boundaries_differ | cached_spikes_differ) {
      f <- tmp$boundaries
      # we'll also compute pct and abs for residual curve (excluding peaks)
      # we'll compute residual curve
      y_res <- tmp$original_curve_y
      # minus spikes
      s <- tmp$spikes
      if (nrow(s) > 0) {
        for(i in 1:nrow(s)) {
          y_res[s$start[i]:s$end[i]] <- 0
        }
      }
      # recompute quantif
      new_res_pct <- r_quantifyFractions(y_res, f)
      new_res_abs <- new_res_pct * tmp$total_protein
      
      # round
      new_res_pct=round(new_res_pct,3)
      new_res_abs=round(new_res_abs,1)
      
      tmp$fractions_residual_qty_pct <- new_res_pct
      tmp$fractions_residual_qty_abs <- new_res_abs
      
      # compute areas under spikes, as well as spikes locations
      if (nrow(s) > 0) {
        spikes_pct <- quantifySpikes(tmp$original_curve_y, s$start, s$end)
        spikes_abs <- spikes_pct * tmp$total_protein
        # round
        spikes_pct=round(spikes_pct,3)
        spikes_abs=round(spikes_abs,1)
        s$qty_pct <- spikes_pct
        s$qty_abs <- spikes_abs
        
        # Compute spike locations
        curve_f <- rep(1, length(tmp$original_curve_y))
        if (length(f)>1) {
          for(i in 1:(length(f)-1)) {
            curve_f[f[i]:(f[i+1]-1)] <- i+1
          }
        }
        if (length(f)>0) {
          curve_f[1:(f[1]-1)] <- 1
          curve_f[f[length(f)]:length(tmp$original_curve_y)] <- length(tmp$fractions_names)
        }
        s_loc <- mapply(function(a,b) {
          curve_f[round((a+b)/2)]
        }, s$start, s$end)
        # Save this for later use
        s$loc <- s_loc
        tmp$spikes <- s
      }
    }
    
    # flags
    if (cached_boundaries_differ | cached_spikes_differ | cached_class_differ | cached_hemolysis_differ) {
      # flags
      # Compute quantities
      frac_qty <- tmp$fractions_qty_abs
      b2g = 0
      if (length(frac_qty) == 6) {
        b2g = sum(frac_qty[5:6])
      }
      if (length(frac_qty) == 5) {
        b2g = frac_qty[5]
        frac_qty <- c(frac_qty[1:4], 0, 0)
      }
      else if (length(frac_qty) == 4) {
        b2g = frac_qty[4]
        frac_qty <- c(frac_qty[1:3], 0, 0, 0)
      }
      else if (length(frac_qty) != 6) {
        frac_qty <- rep(0, 6)
      }
      
      if (length(tmp$fractions_qty_abs) == 6)
        gres <- tmp$fractions_residual_qty_abs[6]
      else
        gres <- NULL
      
      flags <- flagMaker(input_args = list(
        age = tmp$age,
        sex = tmp$sex,
        prot = tmp$total_protein,
        alb = frac_qty[1],
        albq = tmp$computer_analysis$bisalbuminemia_flag,
        a1 = frac_qty[2],
        a2 = frac_qty[3],
        b1 = frac_qty[4],
        b2 = frac_qty[5],
        g = frac_qty[6],
        b2g = b2g,
        class = tmp$class,
        spikes = tmp$spikes$loc,
        frac_names = tmp$fractions_names,
        long_frac_names = tmp$fractions_longnames,
        hemolysis = tmp$haemolysis,
        gres = gres))
      # store & save
      tmp$flags <- flags
      
      # flags text
      # Recompute flags
      printd('Calling flagPrinter with arg:', level=1)
      printd(usedRecommendations$shortname, level=1)
      hr_flags <- flagPrinter(flags, usedRecommendations$shortname)
      # extract human readable text flags
      txt_flags <- hr_flags$prints
      # add red span if warning level > 0
      txt_flags <- paste0(ifelse(hr_flags$warn_lvls == 1, '<span style="color:red">&#8226; ', '&#8226; '), txt_flags, ifelse(hr_flags$warn_lvls == 1, '</span>', ''))
      tmp$cache$flagstext=paste(txt_flags, collapse = '<br>')
    }
    
    # comment
    if (cached_boundaries_differ | cached_spikes_differ | cached_class_differ | cached_hemolysis_differ) {
      commentMaker <- usedRecommendations$fn
      intpt <- commentMaker(input_args = tmp$flags, ant = NULL)
      comment <- intpt$text
      comment <- paste(comment, collapse = '\n')
      tmp$comment <- comment
      tmp$bin_analysis <- intpt$bin
    }
  }
  
  cached_plots <- !is.null(tmp$cache$main_plot)
  
  # finally, outputs : plots
  if (force_plots & (!cached_plots | (cached_boundaries_differ | cached_spikes_differ | cached_cursor_status_differ | cached_class_differ))) {
    # prepare data
    y <- tmp$original_curve_y # curve y
    y_offset_min = 0
    y_offset_max = 1
    y_offset_range = 1
    # select fractions boundaries
    f <- tmp$boundaries
    if (length(f) > 0) {
      f <- data.frame(x = f,
                      y = y[f],
                      ymin = y_offset_min,
                      ymax = y_offset_max,
                      yminus = y_offset_min-y_offset_range/50,
                      name = 1:length(f),
                      stringsAsFactors = F)
      
      f_labels <- tmp$fractions_shortnames
      f_labels <- tmp$fractions_names
      f_labels_x <- c(1, f$x, length(y))
      f_labels_x <- f_labels_x[-length(f_labels_x)]+diff(f_labels_x)/2
      f_labels_y <- y_offset_min-y_offset_range/10
    }
    
    # fill spikes
    pre_p <- tmp$spikes
    p_fill <- data.frame(x = numeric(0),
                         ymin = numeric(0),
                         ymax = numeric(0),
                         clr = numeric(0))
    if (nrow(pre_p) > 0) {
      for(i in 1:nrow(pre_p)) {
        coords = pre_p$start[i]:pre_p$end[i]
        tmp_new_df <- data.frame(x = coords,
                                 ymin = 0,
                                 ymax = y[coords],
                                 clr = pre_p$index[i])
        p_fill <- rbind(p_fill, tmp_new_df)
      }
    }
    p_fill$clr <- as.factor(p_fill$clr)
    
    # spikes names
    p_names <- data.frame(x = numeric(0),
                          y = numeric(0),
                          s = character(0))
    if (nrow(pre_p) > 0) {
      for(i in 1:nrow(pre_p)) {
        # tmp_xp = round((pre_p$start[i]+pre_p$end[i])/2)
        # look for the highest point
        peakzone_y = y[pre_p$start[i]:pre_p$end[i]]
        tmp_xp = which(peakzone_y==max(peakzone_y))[1] + pre_p$start[i] - 1
        tmp_new_df <- data.frame(x = tmp_xp,
                                 y = y[tmp_xp]+.03,
                                 s = paste0(pre_p$index[i]))
        p_names <- rbind(p_names, tmp_new_df)
      }
    }
    
    # compute actual plot
    
    gp <- ggplot()
    
    # choose color of line according to class
    curve_color='black'
    if (tmp$class!=1)
      curve_color='red'
    
    gp <- gp +
      # fill spikes
      geom_ribbon(data = p_fill,
                  aes(x = x, ymin = ymin, ymax = ymax, fill = clr), col = 'red') +
      geom_text(data = p_names,
                aes(x = x, y = y, label = s)) +
      # analysis curve
      geom_line(data = data.frame(x = c(1:length(y)), y = y), aes(x = x, y = y), col = curve_color, size = .5)
    
    if (is.data.frame(f)) {
      gp <- gp +
        # boundaries
        geom_segment(data = f,
                     aes(x = x, y = ymin, xend = x, yend = ymax), col = 'black', size = 1) +
        # boundaries indices
        geom_text(data = f,
                  aes(x = x, y = yminus, label = name), col = 'black', size = 3)
    } else {
      f_tmp <- data.frame(x = 152,
                          yminus = y_offset_min-y_offset_range/50,
                          name = '',
                          stringsAsFactors = F)
      gp <- gp + geom_text(data = f_tmp,
                           aes(x = x, y = yminus, label = name), col = 'black', size = 3)
    }
    
    # if a spike is selected : show it
    if (cursor_status == 1) {
      gp <- gp +
        geom_segment(data = data.frame(x = spike_start, y = 0, yend = 1),
                     aes(x = x, y = y, xend = x, yend = yend), col = 'red', size = 1.25)
    }
    
    if (is.data.frame(f)) {
      gp <- gp +
        # fractions names
        geom_text(data = data.frame(x = f_labels_x, y = f_labels_y, label = f_labels),
                  aes(x = x, y = y, label = label), size = 4)
    } else {
      gp <- gp +
        # fractions names
        geom_text(data = data.frame(x = c(152), y = c(y_offset_min-y_offset_range/10), label = "Not fractionned yet"),
                  aes(x = x, y = y, label = label), size = 4)
    }
    
    gp <- gp +
      # theme
      theme_minimal() +
      theme(axis.line=element_blank(),
            axis.text.x=element_blank(),
            axis.text.y=element_blank(),
            axis.ticks=element_blank(),
            axis.title.x=element_blank(),
            axis.title.y=element_blank(),
            legend.position="none",
            panel.background=element_blank(),
            panel.border=element_blank(),
            panel.grid.major=element_blank(),
            panel.grid.minor=element_blank(),
            plot.background=element_blank())
    
    tmp$cache$main_plot=gp
    
    # Make debug plots
    # Predmaps
    if (T) { # not only DEBUG_MODE
      # Compute fractions predmap
      predmap1=tmp$computer_analysis$predmap_f1
      predmap2=tmp$computer_analysis$predmap_f2
      c=character(length(predmap1))
      c[predmap1==0]='#000000'
      c[predmap1==1]='#59C9A5'
      c[predmap1==2]='#FFC857'
      c[predmap1==3]='#D81E5B'
      c[predmap1==4]='#6DD6DA'
      c[predmap1==5]='#E9724C'
      c[predmap1==6]='#2B59C3'
      if (predmap2[1]>.5) {
        v_offset=-.02
        vjust=1
      } else {
        v_offset=+.02
        vjust=0
      }
      tmp$cache$main_plot_with_fractions_predmap = gp +
        geom_line(data=data.frame(x=c(1:length(y)), y=y), aes(x=x, y=y,group=1), colour=c, size=.75) +
        geom_line(data=data.frame(x=c(1:length(y)), y=predmap2), aes(x=x, y=y), color='#000000',alpha=.8,size=.5) +
        geom_text(data=data.frame(x=1,y=predmap2[1]+v_offset,l=paste0('Mean confidence: ', format(round(mean(predmap2), 3), nsmall = 3))),
                  aes(x=x,y=y,label=l),color='#000000',size=3,alpha=.8,hjust=0,vjust=vjust)
      
      # Compute spikes predmap
      predmap=tmp$computer_analysis$predmap_s1
      # printd('predmap is :')
      # printd(predmap)
      if (predmap[1]>.5) {
        v_offset=-.02
        vjust=1
      } else {
        v_offset=+.02
        vjust=0
      }
      tmp$cache$main_plot_with_spikes_predmap = gp +
        geom_line(data=data.frame(x=c(1:length(y)), y=predmap), aes(x=x, y=y), color='#000000',alpha=.8, size=.5) +
        geom_text(data=data.frame(x=1,y=predmap[1]+v_offset,l=paste0('Max confidence: ', format(round(max(predmap), 3), nsmall = 3))),
                  aes(x=x,y=y,label=l),color='#000000',size=3,alpha=.8,hjust=0,vjust=vjust)
    }
  }
  
  cached_plots <- !is.null(tmp$cache$miniature)
  
  # thumbnail (miniature) plot
  if (force_plots & (!cached_plots | cached_locked_differ)) {
    # recompute small plot
    y <- tmp$original_curve_y # curve y
    # frame:
    # color according to normal rather than locked (not anymore used)
    # frame_color=ifelse(tmp$locked==1,'green','red')
    frame_color=ifelse(tmp$bin_analysis==1,'green',ifelse(tmp$bin_analysis==-1,'red','#333333'))
    # printd(paste0('refreshing miniature for sample: ', tmp$sample_id, ', setting color to:', frame_color))
    # printd('making miniature')
    # printd(paste0('batchid: ',tmp$sample_batchid))
    # printd(paste0('batch index: ',batch_index))
    gp <- ggplot() +
      geom_line(data = data.frame(x = c(1:length(y)), y = y), aes(x = x, y = y), col = '#000000', size = .5) +
      theme_minimal() +
      theme(axis.line=element_blank(),
            axis.text.x=element_blank(),
            axis.text.y=element_blank(),
            axis.ticks=element_blank(),
            # axis.title.x=element_blank(),
            axis.title.y=element_blank(),
            legend.position="none",
            panel.background=element_blank(),
            panel.border = element_rect(colour = frame_color, fill=NA, size=1),
            # panel.border=element_blank(),
            panel.grid.major=element_blank(),
            panel.grid.minor=element_blank(),
            plot.background=element_blank()) +
      xlab(paste0(tmp$sample_id, ' (', batch_index, ')')) +
      coord_fixed(ratio=SPE_WIDTH)
    # printd(gp)
    
    # save plot as image
    fig <- image_graph(width = 200, height=200, res=96)
    print(gp)
    dev.off()
    tmp$cache$miniature=fig
  }

  spe_standards_pct <- ref_values(age=tmp$age, abs=F)
  spe_standards_abs <- ref_values(age=tmp$age, abs=T)
  ref_pct <- ref_values_text(age=tmp$age, abs=F)
  ref_abs <- ref_values_text(age=tmp$age, abs=T)
  
  # tables
  # fractions table
  if (cached_boundaries_differ | cached_spikes_differ) {
    tmp$cache$fractions_dt=NULL
    if (length(tmp$fractions_names) > 0) {
      # pct_v <- round(tmp$fractions_qty_pct*100, 1)
      # abs_v <- round(tmp$fractions_qty_abs, 1)
      pct_v <- tmp$fractions_qty_pct*100
      abs_v <- tmp$fractions_qty_abs
      
      df <- data.frame(tmp1 = tmp$fractions_names,
                       tmp2 = paste0(format(pct_v,nsmall=1), ' %'),
                       tmp3 = rep('-', length(tmp$fractions_names)),
                       tmp4 = paste0(format(abs_v,nsmall=1), ' g/L'),
                       tmp5 = rep('-', length(tmp$fractions_names)),
                       stringsAsFactors = FALSE)
      
      if (nrow(df) == 6) {
        pct_below_lo = pct_v - spe_standards_pct$std_lo < 0
        pct_above_hi = pct_v - spe_standards_pct$std_hi > 0
        abs_below_lo = abs_v - spe_standards_abs$std_lo < 0
        abs_above_hi = abs_v - spe_standards_abs$std_hi > 0
        # add arrow next to text if needed
        # pct_prefix=ifelse(pct_below_lo, '\U02198 ', ifelse(pct_above_hi, '\U02197 ', ''))
        # abs_prefix=ifelse(abs_below_lo, '\U02198 ', ifelse(abs_above_hi, '\U02197 ', ''))
        # set color of text
        colors_pct=ifelse(pct_below_lo, 'orange', ifelse(pct_above_hi, 'red', 'black'))
        colors_abs=ifelse(abs_below_lo, 'orange', ifelse(abs_above_hi, 'red', 'black'))
        # colors_pct = ifelse(pct_below_lo | pct_above_hi, 'red', 'black')
        # colors_abs = ifelse(abs_below_lo | abs_above_hi, 'red', 'black')
        # add ref values
        df[,3] <- ref_pct
        df[,5] <- ref_abs
      } else if (nrow(df) == 5) {
        pct_below_lo = c(pct_v[1:4] - spe_standards_pct$std_lo[1:4] < 0, F)
        pct_above_hi = c(pct_v[1:4] - spe_standards_pct$std_hi[1:4] > 0, F)
        abs_below_lo = c(abs_v[1:4] - spe_standards_abs$std_lo[1:4] < 0, F)
        abs_above_hi = c(abs_v[1:4] - spe_standards_abs$std_hi[1:4] > 0, F)
        # add arrow next to text if needed
        # pct_prefix=ifelse(pct_below_lo, '\U02198 ', ifelse(pct_above_hi, '\U02197 ', ''))
        # abs_prefix=ifelse(abs_below_lo, '\U02198 ', ifelse(abs_above_hi, '\U02197 ', ''))
        # set color of text
        colors_pct=ifelse(pct_below_lo, 'orange', ifelse(pct_above_hi, 'red', 'black'))
        colors_abs=ifelse(abs_below_lo, 'orange', ifelse(abs_above_hi, 'red', 'black'))
        # colors_pct = ifelse(pct_below_lo | pct_above_hi, 'red', 'black')
        # colors_abs = ifelse(abs_below_lo | abs_above_hi, 'red', 'black')
        # add ref values
        df[1:4,3] <- ref_pct[1:4]
        df[1:4,5] <- ref_abs[1:4]
      } else if (nrow(df) == 4) {
        pct_below_lo = c(pct_v[1:3] - spe_standards_pct$std_lo[1:3] < 0, F)
        pct_above_hi = c(pct_v[1:3] - spe_standards_pct$std_hi[1:3] > 0, F)
        abs_below_lo = c(abs_v[1:3] - spe_standards_abs$std_lo[1:3] < 0, F)
        abs_above_hi = c(abs_v[1:3] - spe_standards_abs$std_hi[1:3] > 0, F)
        # add arrow next to text if needed
        # pct_prefix=ifelse(pct_below_lo, '\U02198 ', ifelse(pct_above_hi, '\U02197 ', ''))
        # abs_prefix=ifelse(abs_below_lo, '\U02198 ', ifelse(abs_above_hi, '\U02197 ', ''))
        # set color of text
        colors_pct=ifelse(pct_below_lo, 'orange', ifelse(pct_above_hi, 'red', 'black'))
        colors_abs=ifelse(abs_below_lo, 'orange', ifelse(abs_above_hi, 'red', 'black'))
        # colors_pct = ifelse(pct_below_lo | pct_above_hi, 'red', 'black')
        # colors_abs = ifelse(abs_below_lo | abs_above_hi, 'red', 'black')
        # add ref values
        df[1:3,3] <- ref_pct[1:3]
        df[1:3,5] <- ref_abs[1:3]
      } else {
        # pct_prefix = rep('', nrow(df))
        # abs_prefix = pct_prefix
        colors_pct = rep('black', nrow(df))
        colors_abs = rep('black', nrow(df))
      }
      
      # if there's a spike in gamma-globulins, we'll show residual quantities for gamma
      # we'll just check if residual quantities == quantities for gamma-globulins
      gamma_index <- which(tmp$fractions_names == "Gamma")[1]
      if (!is.na(gamma_index)) { # could happen if bridging
        if (abs(tmp$fractions_qty_abs[gamma_index] - tmp$fractions_residual_qty_abs[gamma_index]) > 1) {
          # tmp_pct <- round(tmp$fractions_residual_qty_pct[gamma_index]*100,1)
          # tmp_abs <- round(tmp$fractions_residual_qty_abs[gamma_index],1)
          tmp_pct <- tmp$fractions_residual_qty_pct[gamma_index]*100
          tmp_abs <- tmp$fractions_residual_qty_abs[gamma_index]
          df[nrow(df)+1, ] <- c('Gamma (res.)',
                                paste0(format(tmp_pct,nsmall=1), ' %'),
                                ref_pct[6],
                                paste0(format(tmp_abs,nsmall=1), ' g/L'),
                                ref_abs[6])
          pct_below_lo = tmp_pct < spe_standards_pct$std_lo[6]
          pct_above_hi = tmp_pct > spe_standards_pct$std_hi[6]
          abs_below_lo = tmp_abs < spe_standards_abs$std_lo[6]
          abs_above_hi = tmp_abs > spe_standards_abs$std_hi[6]
          
          # pct_prefix=c(pct_prefix, ifelse(pct_below_lo, '\U02198 ', ifelse(pct_above_hi, '\U02197 ', '')))
          # abs_prefix=c(abs_prefix, ifelse(abs_below_lo, '\U02198 ', ifelse(abs_above_hi, '\U02197 ', '')))
          colors_pct=c(colors_pct, ifelse(pct_below_lo, 'orange', ifelse(pct_above_hi, 'red', 'black')))
          colors_abs=c(colors_abs, ifelse(abs_below_lo, 'orange', ifelse(abs_above_hi, 'red', 'black')))
          # colors_pct <- c(colors_pct, ifelse(pct_below_lo | pct_above_hi, 'red', 'black'))
          # colors_abs <- c(colors_abs, ifelse(abs_below_lo | abs_above_hi, 'red', 'black'))
        }
      }
      
      # df[,2]=paste0(pct_prefix,df[,2])
      # df[,4]=paste0(pct_prefix,df[,4])
      
      colnames(df) <- c('Fraction', 'Pct.', 'Ref. (%)', 'Conc.', 'Ref. (g/L)')
      
      final_df=datatable(df, rownames = FALSE, options = list(bFilter = FALSE,
                                                              bInfo = FALSE,
                                                              bPaginate = FALSE,
                                                              columnDefs = list(list(className = 'dt-right', targets = c(1, 2, 3, 4))),
                                                              # columnDefs = list(list(
                                                              #   targets = 2,
                                                              #   render = JS(
                                                              #     "function(data, type, row, meta) {",
                                                              #     "return type === 'display' ?",
                                                              #     "data : 'ref';",
                                                              #     "}")
                                                              # )),
                                                              ordering = FALSE)) %>%
        # callback = JS('table.page(0).draw(false);')) %>%
        formatStyle('Pct.', 'Fraction', color = styleEqual(df$Fraction, colors_pct)) %>%
        formatStyle('Conc.', 'Fraction', color = styleEqual(df$Fraction, colors_abs))
      # formatStyle('Pct.', color = styleEqual(df$Pct., colors_pct)) %>%
      # formatStyle('Conc.', color = styleEqual(df$Conc., colors_abs))
      
      # debug
      # printd(paste0('abs_below_lo: ',abs_below_lo))
      # printd(paste0('abs_above_hi: ',abs_above_hi))
      # printd(paste0('abs_v: ',abs_v))
      # printd(paste0('spe_standards_abs: ',spe_standards_abs))
      # printd('making DT:')
      # printd('colors_abs: ',paste(colors_abs,collapse=', '))
      # printd('abs_below_lo: ',paste(abs_below_lo,collapse=', '))
      # printd('abs_below_lo: ',paste(abs_below_lo,collapse=', '))
      tmp$cache$fractions_dt=final_df
    }
  }
  
  # spikes table
  if (cached_boundaries_differ | cached_spikes_differ) {
    tmp$cache$spikes_dt=NULL
    s <- tmp$spikes
    if (nrow(s) > 0) {
      df <- data.frame(tmp1 = 1:nrow(s),
                       tmp2 = paste0(format(s$qty_pct*100,nsmall=1), ' %'),
                       tmp3 = paste0(format(s$qty_abs,nsmall=1), ' g/L'),
                       stringsAsFactors = F)
      colnames(df) <- c('Spike', 'Pct.', 'Conc.')
      final_df = datatable(df, rownames = FALSE, options = list(bFilter = FALSE,
                                                                bInfo = FALSE,
                                                                bPaginate = FALSE,
                                                                columnDefs = list(list(className = 'dt-right', targets = c(1, 2))),
                                                                ordering = FALSE)) %>%
        formatStyle('Conc.', color = 'black')
      tmp$cache$spikes_dt=final_df
    }
  }
  
  printd('post_backend/sampleAnalysis:out')
  
  return(tmp)
}

##################################################################################################
####################################### TEXT BACKEND #############################################
##################################################################################################

# Convert spe analysis results into flags
flagMaker <- function(input_args) {
  printd('text_backend/flagMaker:in')
  
  # input_args = list(age, sex, prot, alb, albq, a1, a2, b1, b2, g, class, class_conf, frac, pic1, pic2, pic3, pic4, pic1_conf, pic2_conf, pic3_conf, pic4_conf, hemolysis, gres)
  # printd("called flagmaker with input:")
  # printd(input_args)
  output_args = list()
  
  # AGE
  output_args$age <- ifelse(input_args$age > 45, 1, 0)
  output_args$ped <- ifelse(input_args$age < 18, 1, 0)
  
  # SEX FLAG
  output_args$sex <- ifelse(input_args$sex == "M", 1, 0)
  
  # PROT FLAG
  if (input_args$prot < 60) {
    if (input_args$prot < 55) {
      output_args$prot <- 2
    } else {
      output_args$prot <- 1
    }
  } else {
    output_args$prot <- 0
  }
  
  ref_df = ref_values(input_args$age, abs=TRUE)
  
  # ALB quantitative FLAG
  if (output_args$ped==1) {
    output_args$alb <- 0
    if (input_args$alb != 0) {
      if (input_args$alb < 5) {
        output_args$alb <- -3
      } else if (input_args$alb < ref_df$std_lo[1]) {
        output_args$alb <- -1
      } else if (input_args$alb > ref_df$std_hi[1]) {
        output_args$alb <- 1
      }
    }
  } else {
    if (input_args$alb != 0) {
      if (input_args$alb <= 50) {
        if (input_args$alb < 35) {
          if (input_args$alb < 30) {
            if (input_args$alb < 5) {
              output_args$alb <- -3
            } else {
              output_args$alb <- -2
            }
          } else {
            output_args$alb <- -1
          }
        } else {
          output_args$alb <- 0
        }
      } else {
        output_args$alb <- 1
      }
    } else {
      output_args$alb <- 0
    }
  }
  
  # ALB qualitative FLAG
  output_args$albq <- input_args$albq
  
  # alpha 1 quantitative FLAG
  if (output_args$ped==1) {
    output_args$a1 <- 0
    if (input_args$a1 != 0) {
      if (input_args$a1 < 1.5) {
        output_args$a1 <- -1
      } else if (input_args$a1 > ref_df$std_hi[2]) {
        output_args$a1 <- 1
      }
    }
  } else {
    if (input_args$a1 != 0) {
      if (input_args$a1 >= 1.5) {
        if (input_args$a1 > 4) {
          if (input_args$a1 > 6) {
            output_args$a1 <- 2
          } else {
            output_args$a1 <- 1
          }
        } else {
          output_args$a1 <- 0
        }
      } else {
        output_args$a1 <- -1
      }
    } else {
      output_args$a1 <- 0
    }
  }
  
  
  # Alpha2 quantitative FLAG
  if (output_args$ped==1) {
    output_args$a2 <- 0
    if (input_args$a2 > ref_df$std_hi[3]) {
      output_args$a2 <- 1
    }
  } else {
    if (input_args$a2 > 9) {
      if (input_args$a2 > 12) {
        output_args$a2 <- 2
      } else {
        output_args$a2 <- 1
      }
    } else {
      output_args$a2 <- 0
    }
  }
  
  # Beta1 quantitative FLAG
  if (output_args$ped==1) {
    output_args$b1 <- 0
    if (input_args$b1 > ref_df$std_hi[4]) {
      output_args$b1 <- 1
    }
  } else {
    if (input_args$b1 > 6) {
      output_args$b1 <- 1
    } else {
      output_args$b1 <- 0
    }
  }
  
  # Beta2 quantitative FLAG
  if (output_args$ped==1) {
    output_args$b2 <- 0
    if (input_args$b2 != 0) {
      if (input_args$b2 < ref_df$std_lo[5]) {
        output_args$b2 <- -1
      } else if (input_args$b2 > ref_df$std_hi[5]) {
        output_args$b2 <- 1
      }
    }
  } else {
    if (input_args$b2 != 0) {
      if (input_args$b2 >= 2) {
        if (input_args$b2 > 5.5) {
          if (input_args$b2 > 8) {
            output_args$b2 <- 2
          } else {
            output_args$b2 <- 1
          }
        } else {
          output_args$b2 <- 0
        }
      } else {
        output_args$b2 <- -1
      }
    } else {
      output_args$b2 <- 0
    }
  }
  
  # Beta2 > beta1 quantitative FLAG
  if (output_args$ped==0 & input_args$b2 != 0 & input_args$b1 != 0 & input_args$b2 > input_args$b1) {
    output_args$b2b1 <- 1
  } else {
    output_args$b2b1 <- 0
  }
  
  # input_args$gamma quantitative FLAG
  if (output_args$ped==1) {
    output_args$g <- 0
    if (input_args$g != 0) {
      if (input_args$g < ref_df$std_lo[6]) {
        output_args$g <- -1
      } else if (input_args$g > ref_df$std_hi[6]) {
        output_args$g <- 1
      }
    }
  } else {
    if (input_args$g != 0) {
      if (input_args$g >= 5) {
        if (input_args$g >= 8) {
          if (input_args$g > 15) {
            if (input_args$g > 20) {
              output_args$g <- 2
            } else {
              output_args$g <- 1
            }
          } else {
            output_args$g <- 0
          }
        } else {
          output_args$g <- -1
        }
      } else {
        output_args$g <- -2
      }
    } else {
      output_args$g <- 0
    }
  }
  
  # GAMMA PRECISE (for MOSS 2016)
  # input_args$gamma quantitative FLAG
  if (output_args$ped==1) {
    output_args$g_prec <- 0
    if (input_args$g != 0) {
      if (input_args$g < ref_df$std_lo[6]) {
        output_args$g_prec <- -1
      } else if (input_args$g > ref_df$std_hi[6]) {
        output_args$g_prec <- 1
      }
    }
  } else {
    if (input_args$g != 0) {
      if (input_args$g < 5) {
        output_args$g_prec <- -4
      } else if (input_args$g >= 5 & input_args$g < 6) {
        output_args$g_prec <- -3
      } else if (input_args$g >= 6 & input_args$g < 7) {
        output_args$g_prec <- -2
      } else if (input_args$g >= 7 & input_args$g < 8) {
        output_args$g_prec <- -1
      } else if (input_args$g >= 8 & input_args$g <= 15) {
        output_args$g_prec <- 0
      } else if (input_args$g > 15 & input_args$g <= 16) {
        output_args$g_prec <- 1
      } else if (input_args$g > 16 & input_args$g <= 18) {
        output_args$g_prec <- 2
      } else if (input_args$g > 18 & input_args$g <= 20) {
        output_args$g_prec <- 3
      } else if (input_args$g > 20) {
        output_args$g_prec <- 4
      }
    } else {
      output_args$g_prec <- 0
    }
  }
  
  # check fractions names and number
  n_frac <- length(input_args$frac_names)
  if (n_frac > 0) {
    if (length(input_args$frac_names) == n_frac & length(input_args$long_frac_names) == n_frac) {
      output_args$frac_names <- tolower(input_args$frac_names)
      output_args$long_frac_names <- tolower(input_args$long_frac_names)
      output_args$n_frac <- n_frac
    } else {
      output_args$frac_names <- rep("?", n_frac)
      output_args$long_frac_names <- rep("?", n_frac)
      output_args$n_frac <- n_frac
    }
  } else {
    n_frac <- 1
    output_args$frac_names <- c("?")
    output_args$long_frac_names <- c("?")
    output_args$n_frac <- n_frac
  }
  
  # QUalitative FLAG : CLASSIFIER
  output_args$class = input_args$class
  output_args$subclass = 0
  if (input_args$class == 4){
    if (input_args$b2g > 20){
      if (input_args$b2g > 30){
        output_args$subclass = 2
      } else {
        output_args$subclass = 1
      }
    }
  }
  
  # Spikes
  spikes <- input_args$spikes
  # check all spikes match stored fractions
  spikes[spikes>n_frac] <- 1
  output_args$spikes <- spikes
  output_args$n_spikes <- length(spikes)
  
  output_args$hemolysis <- input_args$hemolysis
  
  # RESIDUAL GAMMA
  if (!is.null(input_args$gres)) {
    if(abs(input_args$gres-input_args$g) > 1) {
      gres <- input_args$gres
      if (output_args$ped==1) {
        if (gres < ref_df$std_lo[6]) {
          output_args$gres <- -1
          output_args$gres_prec <- -1
        } else if (gres > ref_df$std_hi[6]) {
          output_args$gres <- 1
          output_args$gres_prec <- 1
        } else {
          output_args$gres <- 0
          output_args$gres_prec <- 0
        }
      } else {
        if (gres < 8) {
          output_args$gres <- -1
        } else if (gres > 15) {
          output_args$gres <- 1
        } else {
          output_args$gres <- 0
        }
        
        # MOSS 2016
        if (gres < 5) {
          output_args$gres_prec <- -4
        } else if (gres >= 5 & gres < 6) {
          output_args$gres_prec <- -3
        } else if (gres >= 6 & gres < 7) {
          output_args$gres_prec <- -2
        } else if (gres >= 7 & gres < 8) {
          output_args$gres_prec <- -1
        } else if (gres >= 8 & gres <= 15) {
          output_args$gres_prec <- 0
        } else if (gres > 15 & gres <= 16) {
          output_args$gres_prec <- 1
        } else if (gres > 16 & gres <= 18) {
          output_args$gres_prec <- 2
        } else if (gres > 18 & gres <= 20) {
          output_args$gres_prec <- 3
        } else if (gres > 20) {
          output_args$gres_prec <- 4
        } else {
          output_args$gres_prec <- 0
        }
      }
    } else {
      output_args$gres <- NULL
      output_args$gres_prec <- NULL
    }
  }
  
  # PLASMAPHERESIS FLAG
  # plasmapheresis = albumin > 30, a1, a2, b1, b2, g decreased or peak in one of those fractions
  corr_fac <- .5 # decreased = below (referefence value+0.5g/L)
  if (input_args$alb > 30 &
      (input_args$a1 < ref_df$std_lo[2]+corr_fac | sum(spikes==2) > 0) &
      (input_args$a2 < ref_df$std_lo[3]+corr_fac | sum(spikes==3) > 0) &
      (input_args$b1 < ref_df$std_lo[4]+corr_fac | sum(spikes==4) > 0) &
      (input_args$b2 < ref_df$std_lo[5]+corr_fac | sum(spikes==5) > 0) &
      (input_args$g < ref_df$std_lo[6]+corr_fac | sum(spikes==6) > 0)) {
    output_args$plasmapheresis = 1
  } else {
    output_args$plasmapheresis = 0
  }
  
  # NEPHROTIC SYNDROME FLAG
  if (input_args$prot < 60 &
      input_args$alb < 30 &
      input_args$a1 <= 4 &
      input_args$a2 > 11) {
    output_args$nephrotic = 1
  } else {
    output_args$nephrotic = 0
  }
  
  printd('text_backend/flagMaker:out')
  
  # creating the flags named vector to store the flags 
  # printd('returning flags:')
  # printd(output_args)
  output_args
}

# Function to print the flags in human readable comment
flagPrinter <- function(input_args, used_recommendations){
  printd('text_backend/flagPrinter:in')
  
  if (!is.list(input_args))
    return(NULL)
  
  # vector to store the flags comment
  # prints <- c(rep(NA, 13+length(frac_names)))
  # warn_lvls <- c(rep(0, 13+length(frac_names)))
  prints <- c(rep(NA, 15+length(input_args$spikes)))
  warn_lvls <- c(rep(0, 15+length(input_args$spikes)))
  
  n <- 1
  if (input_args$class == 4){
    if (used_recommendations=="SZYM") {
      if (input_args$subclass == 2) {
        prints[n] <- paste0("Beta-gamma bridging, b2+gamma > 30g/l")
        warn_lvls[n] <- 1
      } else if (input_args$subclass == 1) {
        prints[n] <- paste0("Beta-gamma bridging, b2+gamma > 20g/l")
        warn_lvls[n] <- 1
      } else {
        prints[n] <- paste("Beta-gamma bridging")
        warn_lvls[n] <- 1
      }
    } else if (used_recommendations=="MOSS") {
      if (input_args$b1 > 0 | input_args$b2 > 0) {
        prints[n] <- paste0("Beta-gamma bridging, b1/b2 inc.")
        warn_lvls[n] <- 1
      } else {
        prints[n] <- paste("Beta-gamma bridging")
        warn_lvls[n] <- 1
      }
    }
  } else if (input_args$class == 2){
    prints[n] <- paste("Heterogeneity restriction aspect")
    warn_lvls[n] <- 1
  } else if (input_args$class == 3){
    prints[n] <- paste("Clonal anomaly")
    warn_lvls[n] <- 1
  }
  
  n <- n+1
  if (input_args$age == 1){
    prints[n] <- "Age > 45" # ≥
  } else if (input_args$ped == 1) {
    prints[n] <- "Age < 18 (pediatric reference values)"
  }
  
  n <- n+1
  if (input_args$sex == 1){
    prints[n] <- "Male"
  } else {
    prints[n] <- "Female"
  }
  
  n <- n+1
  if (input_args$plasmapheresis == 1){
    prints[n] <- "Plasmapheresis?"
    warn_lvls[n] <- 1
  }
  
  n <- n+1
  if (input_args$nephrotic == 1){
    prints[n] <- "Nephrotic syndrome?"
    warn_lvls[n] <- 1
  }
  
  n <- n+1
  if (input_args$prot == 1){
    prints[n] <- "Total protein < 60 g/l"
  } else if (input_args$prot == 2){
    prints[n] <- "Total protein < 55 g/l"
  }
  
  n <- n+1
  if (input_args$ped == 1) {
    if (input_args$alb == 1){
      prints[n] <- "Albumin > ref (g/l)"
    } else if (input_args$alb == -1){
      prints[n] <- "Albumin < ref (g/l)"
    } else if (input_args$alb == -3){
      prints[n] <- "Albumin < 5 g/l"
      warn_lvls[n] <- 1
    }
  } else {
    if (input_args$alb == 1){
      prints[n] <- "Albumin > 50 g/l"
    } else if (input_args$alb == -1){
      prints[n] <- "Albumin < 35 g/l"
    } else if (input_args$alb == -2){
      prints[n] <- "Albumin < 30 g/l"
    } else if (input_args$alb == -3){
      prints[n] <- "Albumin < 5 g/l"
      warn_lvls[n] <- 1
    }
  }
  
  n <- n+1
  if (input_args$albq == 1){
    prints[n] <- "Bisalbuminemia"
    warn_lvls[n] <- 1
  }
  
  n <- n+1
  if (input_args$ped == 1) {
    if (input_args$a1 == -1){
      prints[n] <- "Alpha-1-globulins < 1.5 g/l"
      warn_lvls[n] <- 1
    } else if (input_args$a1 == 1){
      prints[n] <- "Alpha-1-globulins > ref (g/l)"
    }
  } else {
    if (input_args$a1 == -1){
      prints[n] <- "Alpha-1-globulins < 1.5 g/l"
      warn_lvls[n] <- 1
    } else if (input_args$a1 == 1){
      prints[n] <- "Alpha-1-globulins > 4 g/l"
    } else if (input_args$a1 == 2){
      prints[n] <- "Alpha-1-globulins > 6 g/l"
    }
  }
  
  n <- n+1
  if (input_args$ped == 1) {
    if (input_args$a2 == 1){
      prints[n] <- "Alpha-2-globulins > ref (g/l)"
    }
  } else {
    if (input_args$a2 == 1){
      prints[n] <- "Alpha-2-globulins > 9 g/l"
    } else if (input_args$a2 == 2){
      prints[n] <- "Alpha-2-globulins > 12 g/l"
    }
  }
  
  n <- n+1
  if (input_args$ped == 1) {
    if (input_args$b1 == 1){
      prints[n] <- "Beta-1-globulins > ref (g/l)"
    }
  } else {
    if (input_args$b1 == 1){
      prints[n] <- "Beta-1-globulins > 6 g/l"
    }
  }
  
  n <- n+1
  if (input_args$ped == 1) {
    if (input_args$b2 == -1){
      prints[n] <- "Beta-2-globulins < ref (g/l)"
    } else if (input_args$b2 == 1){
      prints[n] <- "Beta-2-globulins > ref (g/l)"
    }
  } else {
    if (input_args$b2 == -1){
      prints[n] <- "Beta-2-globulins < 2 g/l"
    } else if (input_args$b2 == 1){
      prints[n] <- "Beta-2-globulins > 5.5 g/l"
    } else if (input_args$b2 == 2){
      prints[n] <- "Beta-2-globulins > 8 g/l"
    }
  }
  
  n <- n+1
  if (input_args$b2b1 == 1){
    prints[n] <- "Beta-2-globulins > beta-1-globulins"
  }
  
  n <- n+1
  if (used_recommendations == "SZYM") {
    if (input_args$ped == 1) {
      if (is.null(input_args$gres)) {
        g_value <- input_args$g
      } else {
        g_value <- input_args$gres
      }
      if (g_value == -1){
        prints[n] <- "Gammaglobulins < ref (g/l)"
        warn_lvls[n] <- 1
      } else if (g_value == 1){
        prints[n] <- "Gammaglobulins > ref (g/l)"
        warn_lvls[n] <- 1
      }
    } else {
      if (is.null(input_args$gres)) {
        if (input_args$g == -2){
          prints[n] <- "Gammaglobulins < 5 g/l"
          warn_lvls[n] <- 1
        } else if (input_args$g == -1){
          prints[n] <- "Gammaglobulins < 8 g/l"
          warn_lvls[n] <- 1
        } else if (input_args$g == 1){
          prints[n] <- "Gammaglobulins > 15 g/l"
          warn_lvls[n] <- 1
        } else if (input_args$g == 2){
          prints[n] <- "Gammaglobulins > 20 g/l"
          warn_lvls[n] <- 1
        }
      } else {
        if (input_args$gres == -1){
          prints[n] <- "Gamma (res) < 8 g/l"
          warn_lvls[n] <- 1
        } else if (input_args$gres == 1){
          prints[n] <- "Gamma (res) > 15 g/l"
          warn_lvls[n] <- 1
        }
      }
    }
  } else if (used_recommendations == "MOSS") {
    if (input_args$ped == 1) {
      if (is.null(input_args$gres)) {
        g_value <- input_args$g
      } else {
        g_value <- input_args$gres
      }
      if (g_value == -1){
        prints[n] <- "Gammaglobulins < ref (g/l)"
        warn_lvls[n] <- 1
      } else if (g_value == 1){
        prints[n] <- "Gammaglobulins > ref (g/l)"
        warn_lvls[n] <- 1
      }
    } else {
      if (is.null(input_args$gres_prec)) {
        g_value <- input_args$g_prec
        g_name <- "Gammaglobulins"
      } else {
        g_value <- input_args$gres_prec
        g_name <- "Gamma (res)"
      }
      if (g_value == -4){
        prints[n] <- paste(g_name, "< 5 g/l")
        warn_lvls[n] <- 1
      } else if (g_value == -3){
        prints[n] <- paste(g_name, "5-6 g/l")
        warn_lvls[n] <- 1
      } else if (g_value == -2){
        prints[n] <- paste(g_name, "6-7 g/l")
        warn_lvls[n] <- 1
      } else if (g_value == -1){
        prints[n] <- paste(g_name, "7-8 g/l")
        warn_lvls[n] <- 1
      } else if (g_value == 1){
        prints[n] <- paste(g_name, "15-16 g/l")
        warn_lvls[n] <- 1
      } else if (g_value == 2){
        prints[n] <- paste(g_name, "16-18 g/l")
        warn_lvls[n] <- 1
      } else if (g_value == 3){
        prints[n] <- paste(g_name, "18-20 g/l")
        warn_lvls[n] <- 1
      } else if (g_value == 4){
        prints[n] <- paste(g_name, "> 20 g/l")
        warn_lvls[n] <- 1
      }
    }
  }

  if (input_args$n_spikes > 0) {
    spikes_loc <- input_args$spikes
    spikes_loc <- spikes_loc[order(spikes_loc)]
    for (i in 1:length(input_args$frac_names)) {
      n <- n+1
      n_spikes_in_frac <- sum(spikes_loc==i)
      if (n_spikes_in_frac > 0) {
        prints[n] <- paste0(n_spikes_in_frac, " spike(s) in ", input_args$frac_names[i])
        warn_lvls[n] <- 1
      }
    }
  }
  
  # removing the NA to get a vector with our flag comments only
  keep <- !is.na(prints)
  prints <- prints[keep]
  warn_lvls <- warn_lvls[keep]
  
  printd('text_backend/flagPrinter:out')
  
  return(list(prints = prints, warn_lvls = warn_lvls))
}

#####
# SYMANOWICS
#####

commentMaker_SZYM2006 <- function(input_args, ant=NULL){
  printd('text_backend/commentMaker:in')

  if (input_args$n_frac < 4 | input_args$n_frac > 6) {
    printd('text_backend/commentMaker:out(null)')
    if (LOCALIZATION=='FR') {
      return(list(text="Interprétation impossible.", bin=0))
    } else if (LOCALIZATION=='EN') {
      return(list(text="Interpretation impossible.", bin=0))
    }
  }

  # vector to store the flags comment
  comment <- c(rep(NA, 44))
  qtyqly_normal <- -1

  # Conditions for each comments
  if (input_args$prot == 0 & input_args$alb== 0 & input_args$albq== 0 & input_args$a1 == 0 & input_args$a2 == 0 & input_args$b1 == 0
      & input_args$b2 == 0 & input_args$g == 0 & input_args$class == 1 & input_args$nephrotic == 0 & input_args$plasmapheresis == 0){
    qtyqly_normal <- 1
    if (LOCALIZATION=='FR')
      comment[12] <- "Profil qualitatif et quantitatif de l'électrophorèse sans anomalie notable."
    if (LOCALIZATION=='EN')
      comment[12] <- "Qualitative and quantitative electrophoresis profile without significant abnormalities."
  }

  # if (input_args$prot == 0 & input_args$alb== 0 & input_args$albq== 0 & input_args$a1 == 0 & input_args$a2 == 0 & input_args$b1 == 0
  #     & input_args$b2 == 1 & input_args$g == 0 & input_args$class == 1 & input_args$nephrotic == 0 & input_args$plasmapheresis == 0){
  #   qtyqly_normal <- 1
  #   if (LOCALIZATION=='FR')
  #     comment[12] <- "Profil qualitatif et quantitatif de l'électrophorèse sans anomalie notable."
  #   if (LOCALIZATION=='EN')
  #     comment[12] <- "Qualitative and quantitative electrophoresis profile without significant abnormalities."
  # }

  if (input_args$nephrotic==1) {
    if (LOCALIZATION=='FR')
      comment[8] <- "Profil en faveur d'un syndrome néphrotique. À confronter avec le contexte clinique. À confirmer par le dosage de la protéinurie des 24 heures."
    if (LOCALIZATION=='EN')
      comment[8] <- "Profile in favor of a nephrotic syndrome. To be compared with the clinical context and 24-hour urine protein assay."
  }

  if (input_args$plasmapheresis==1){
    if (LOCALIZATION=='FR')
      comment[10] <- "Diminution globale des globulines : échange plasmatiques?"
    if (LOCALIZATION=='EN')
      comment[10] <- "Global decrease of globulins: plasmapheresis?"
  }

  if (input_args$hemolysis == 1){
    if (LOCALIZATION=='FR')
      comment[11] <- "Sérum hémolysé, résultats sous réserve. Renvoyer un échantillon pour contrôle si nécessaire."
    if (LOCALIZATION=='EN')
      comment[11] <- "Hemolyzed serum, results are subject to misinterpretation. Results may be checked using a new sample."
  }

  global_hypoprot = F
  if (input_args$prot == 2 & input_args$alb < 1 & input_args$albq== 0
      & input_args$a1 < 1 & input_args$a2 == 0 & input_args$b1 == 0
      & input_args$b2 < 1 & input_args$g < 1
      & input_args$class == 1 & !input_args$plasmapheresis & !input_args$nephrotic){
    global_hypoprot = T
    if (LOCALIZATION=='FR')
      comment[10] <- "Hypoprotéinémie globale (fuite urinaire ou digestive, insuffisance hépatique, dénutrition, dilution)."
    if (LOCALIZATION=='EN')
      comment[10] <- "Global hypoproteinemia (urinary or digestive leakage, liver failure, undernutrition, dilution)."
  }

  if (!input_args$plasmapheresis & !input_args$nephrotic) {
    if (input_args$ped==1) {
      if (input_args$alb > -1
          & (input_args$a1 == 1 | input_args$a2 > 0)){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil compatible avec un syndrome inflammatoire."
        if (LOCALIZATION=='EN')
          comment[6] <- "Profile compatible with an inflammatory syndrome."
      }

      if (input_args$alb == -1
          & (input_args$a1 == 1 | input_args$a2 > 0 )){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil compatible avec un syndrome inflammatoire et diminution de l'albumine."
        if (LOCALIZATION=='EN')
          comment[6] <- "Profile compatible with an inflammatory syndrome and hypoalbuminemia."
      }

      if ((input_args$a1 == 1 | input_args$a2 > 0) & input_args$g == 1 & input_args$class==1){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil électrophorétique compatible avec un syndrome inflammatoire accompagné d'une réaction immunitaire."
        if (LOCALIZATION=='EN')
          comment[6] <- "Profile compatible with an inflammatory syndrome with immune response."
      }

      if (input_args$alb == -1 & input_args$a1 == 0 & input_args$a2 == 0 & !global_hypoprot){
        if (LOCALIZATION=='FR')
          comment[7] <- "Hypoalbuminémie"
        if (LOCALIZATION=='EN')
          comment[7] <- "Hypoalbuminemia."
      }
    } else {
      if (input_args$alb > -1
          & (input_args$a1 == 1 | input_args$a2 == 1)){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil compatible avec un syndrome inflammatoire modéré."
        if (LOCALIZATION=='EN')
          comment[6] <- "Profile compatible with a moderate inflammatory syndrome."
      }

      if (input_args$alb == -1
          & (input_args$a1 == 1 | input_args$a2 == 1)){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil compatible avec un syndrome inflammatoire modéré et diminution de l'albumine."
        if (LOCALIZATION=='EN')
          comment[6] <- "Profile compatible with a moderate inflammatory syndrome and hypoalbuminemia."
      }

      if (input_args$alb == -2
          & (input_args$a1 == 1 | input_args$a2 == 1)){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil compatible avec un syndrome inflammatoire modéré et diminution importante de l'albumine."
        if (LOCALIZATION=='EN')
          comment[6] <- "Profile compatible with a moderate inflammatory syndrome and major hypoalbuminemia."
      }

      if (input_args$alb > -1
          & (input_args$a1 == 2 | input_args$a2 == 2)){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil compatible avec un syndrome inflammatoire important."
        if (LOCALIZATION=='EN')
          comment[6] <- "Profile compatible with a major inflammatory syndrome."
      }

      if (input_args$alb == -1
          & (input_args$a1 == 2 | input_args$a2 == 2)){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil compatible avec un syndrome inflammatoire important et diminution de l'albumine."
        if (LOCALIZATION=='EN')
          comment[6] <- "Profile compatible with a major inflammatory syndrome and hypoalbuminemia."
      }

      if (input_args$alb == -2
          & (input_args$a1 == 2 | input_args$a2 == 2)){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil compatible avec un syndrome inflammatoire important et diminution importante de l'albumine."
        if (LOCALIZATION=='EN')
          comment[6] <- "Profile compatible with a major inflammatory syndrome and major hypoalbuminemia."
      }

      if ((input_args$a1 > 0 | input_args$a2 > 0) & input_args$g > 0 & input_args$class==1){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil électrophorétique compatible avec un syndrome inflammatoire accompagné d'une réaction immunitaire."
        if (LOCALIZATION=='EN')
          comment[6] <- "Profile compatible with an inflammatory syndrome with immune response."
      }

      if (input_args$alb == -1 & input_args$a1 == 0 & input_args$a2 == 0 & !global_hypoprot){
        if (LOCALIZATION=='FR')
          comment[7] <- "Hypoalbuminémie modérée."
        if (LOCALIZATION=='EN')
          comment[7] <- "Moderate hypoalbuminemia."
      }

      if (input_args$alb == -2 & input_args$a1 == 0 & input_args$a2 == 0 & !global_hypoprot){
        if (LOCALIZATION=='FR')
          comment[7] <- "Hypoalbuminémie importante."
        if (LOCALIZATION=='EN')
          comment[7] <- "Major hypoalbuminemia."
      }
    }

    if (input_args$prot == 0 & input_args$a1 == -1){
      if (LOCALIZATION=='FR')
        comment[9] <- "Diminution importante de la zone des alpha-1-globulines, compatible avec un déficit en alpha-1-antitrypsine."
      if (LOCALIZATION=='EN')
        comment[9] <- "Major decrease of alpha-1-globulins, compatible with alpha-1-antitrypsin deficiency."
    }
  }

  if (input_args$prot == 0 & input_args$alb== 1){
    if (LOCALIZATION=='FR')
      comment[7] <- "Augmentation de l'albumine, hémoconcentration très probable. À confronter au contexte clinique."
    if (LOCALIZATION=='EN')
      comment[7] <- "Increase of albumin, probable hemoconcentration. To be compared with the clinical context."
  }

  if (input_args$albq== 1){
    if (LOCALIZATION=='FR')
      comment[8] <- "Bisalbuminémie soit congénitale (variant de l'albumine) soit secondaire à un traitement par des bêtalactamines (pénicillines ou céphalosporines), ou une protéolyse par les enzymes pancréatiques libérées en excès. Évolution à surveiller si nécessaire."
    if (LOCALIZATION=='EN')
      comment[8] <- "Bisalbuminemia, either congenital (albumin variant) or secondary to a treatment by beta-lactamins or proteolysis by pancreatic enzymes. Evolution to be monitored."
  }

  if (input_args$alb== -3){
    if (LOCALIZATION=='FR')
      comment[9] <-  "Absence d'albumine : analbuminémie congénitale avec augmentation de toute les fractions globuliniques."
    if (LOCALIZATION=='EN')
      comment[9] <-  "Absence of albumin: congenital analbuminemia with increase of all globulins."
  }

  if (input_args$age== 1 & input_args$b2b1 == 1 & input_args$g == -2 & input_args$class == 1 & !input_args$plasmapheresis & !input_args$nephrotic){
    if (LOCALIZATION=='FR')
      comment[5] <-  "Les bêta-2-globulines sont supérieures aux bêta-1-globulines avec diminution des gammaglobulines. L'identification immunologique est recommandée selon le contexte clinique."
    if (LOCALIZATION=='EN')
      comment[5] <-  "Beta-2-globulins are higher than beta-1-globulins, in a context of hypogammaglobulinemia. Immunological identification is recommended as appropriate to the clinical context."
  }

  if (input_args$class == 4 & input_args$subclass == 0){
    if (LOCALIZATION=='FR')
      comment[4] <-  "Bloc bêta-gamma débutant."
    if (LOCALIZATION=='EN')
      comment[4] <-  "Early beta-gamma bridging."
  }

  if (input_args$class == 4 & input_args$subclass == 1){
    if (LOCALIZATION=='FR')
      comment[4] <-  "Bloc bêta-gamma important."
    if (LOCALIZATION=='EN')
      comment[4] <-  "Major beta-gamma bridging."
  }

  if (input_args$class == 4 & input_args$subclass == 2){
    if (LOCALIZATION=='FR')
      comment[4] <-  "Bloc bêta-gamma avec augmentation polyclonale importante des immunoglobulines."
    if (LOCALIZATION=='EN')
      comment[4] <-  "Beta-gamma bridging with major polyclonal increase of gammaglobulins."
  }

  if (input_args$sex == 0 & input_args$b1 == 1 & input_args$class == 1){
    if (LOCALIZATION=='FR')
      comment[13] <-  "Augmentation des bêta-1-globulines (transferrine) compatible avec une sidéropénie ou une imprégnation oestogénique. À confronter au contexte clinique. Évolution à surveiller si inexpliquée."
    if (LOCALIZATION=='EN')
      comment[13] <-  "Increase of beta-1-globulins (transferrin) compatible with sideropenia or estrogen impregnation. Evolution to be monitored as appropriate to the clinical context."
  }

  if (input_args$sex == 1 & input_args$b1 == 1 & input_args$class == 1){
    if (LOCALIZATION=='FR')
      comment[13] <-  "Augmentation des bêta-1-globulines (transferrine) compatible avec une sidéropénie. À confronter au contexte clinique. Évolution à surveiller si inexpliquée."
    if (LOCALIZATION=='EN')
      comment[13] <-  "Increase of beta-1-globulins (transferrin) compatible with sideropenia. Evolution to be monitored as appropriate to the clinical context."
  }

  # if (input_args$a1 <= 0 & input_args$a2 <= 0 & input_args$b2 == 1 & input_args$class == 1){
  #   if (LOCALIZATION=='FR')
  #     comment[5] <-   "Augmentation modérée des bêta-2-globulines. Évolution à surveiller en fonction du contexte clinique."
  #   if (LOCALIZATION=='EN')
  #     comment[5] <-   "Moderate increase of beta-2-globulins. Evolution to be monitored as appropriate to the clinical context."
  # }

  # ne sera pas donné pour péd (b2>1)
  if (input_args$a1 <= 0 & input_args$a2 <= 0 & input_args$b2 == 2 & input_args$class == 1){
    if (LOCALIZATION=='FR')
      comment[5] <-   "Augmentation importante des bêta-2-globulines. Évolution à surveiller en fonction du contexte clinique."
    if (LOCALIZATION=='EN')
      comment[5] <-   "Major increase of beta-2-globulins. Evolution to be monitored as appropriate to the clinical context."
  }

  if (input_args$b2 == -1 & !input_args$plasmapheresis & !input_args$nephrotic){
    if (LOCALIZATION=='FR')
      comment[14] <-   "Diminution importante des bêta-2-globulines consécutive à l'activation de la voie alterne et ou classique du complément. À confronter au contexte clinique."
    if (LOCALIZATION=='EN')
      comment[14] <-   "Major decrease of beta-2-globulins, which may be due to the activation of the complement pathway. To be compared with the clinical context."
  }

  if (!input_args$plasmapheresis & !input_args$nephrotic) {
    if (input_args$ped==1) {
      if (input_args$g == -1 & input_args$class == 1){
        if (LOCALIZATION=='FR')
          comment[4] <-   "Diminution des gammaglobulines."
        if (LOCALIZATION=='EN')
          comment[4] <-   "Decrease of gammaglobulins."
      }

      if (input_args$g == 1 & input_args$class == 1){
        if (LOCALIZATION=='FR')
          comment[4] <-   "Augmentation polyclonale des gammaglobulines."
        if (LOCALIZATION=='EN')
          comment[4] <-   "Polyclonal increase of gammaglobulins."
      }

      if (input_args$g == 0 & input_args$class == 2){
        if (LOCALIZATION=='FR')
          comment[4] <-  "Aspect de restriction d'hétérogénéité des gammaglobulines."
        if (LOCALIZATION=='EN')
          comment[4] <-  "Restriction of immunoglobulin heterogeneity."
      }

      if (input_args$g == -1 & input_args$class == 2){
        if (LOCALIZATION=='FR')
          comment[4] <-   "Aspect de restriction d'hétérogénéité dans un contexte d'hypogammaglobulinémie."
        if (LOCALIZATION=='EN')
          comment[4] <-   "Restriction of immunoglobulin heterogeneity with hypogammaglobulinemia."
      }

      if (input_args$g == 1 & input_args$class == 2){
        if (LOCALIZATION=='FR')
          comment[4] <-  "Aspect de restriction d'hétérogénéité des gammaglobulines dans un contexte d'hypergammaglobulinémie."
        if (LOCALIZATION=='EN')
          comment[4] <-  "Restriction of immunoglobulin heterogeneity with hypergammaglobulinemia."
      }
    } else {
      if (input_args$g == -1 & input_args$age == 1 & input_args$class == 1){
        if (LOCALIZATION=='FR')
          comment[4] <-   "Discrète diminution des gammaglobulines. Évolution à surveiller en fonction du contexte clinique."
        if (LOCALIZATION=='EN')
          comment[4] <-   "Minor decrease of gammaglobulins. Evolution to be monitored as appropriate to the clinical context."
      }

      if (input_args$g == -1 & input_args$age == 0 & input_args$class == 1){
        if (LOCALIZATION=='FR')
          comment[4] <-   "Discrète diminution des gammaglobulines."
        if (LOCALIZATION=='EN')
          comment[4] <-   "Minor decrease of gammaglobulins."
      }

      if (input_args$g == 1 & input_args$class == 1){
        if (LOCALIZATION=='FR')
          comment[4] <-  "Augmentation polyclonale modérée des gammaglobulines."
        if (LOCALIZATION=='EN')
          comment[4] <-  "Moderate polyclonal increase of gammaglobulins."
      }

      if (input_args$g == 2 & input_args$class == 1){
        if (LOCALIZATION=='FR')
          comment[4] <-  "Augmentation polyclonale importante des gammaglobulines."
        if (LOCALIZATION=='EN')
          comment[4] <-  "Major polyclonal increase of gammaglobulins."
      }

      if (input_args$g == -2 & input_args$class == 1){
        if (LOCALIZATION=='FR')
          comment[4] <-  "Importante diminution des gammaglobulines. L'identification immunologique sérique, le dosage des chaînes légères libres sériques et l'électrophorèse des protéines urinaires sont recommandés selon le contexte clinique."
        if (LOCALIZATION=='EN')
          comment[4] <-  "Major decrease of gammaglobulins. Serum immunological identification, free light chains assay and urine protein electrophoresis are recommended as appropriate to the clinical context."
      }

      if (input_args$g == 0 & input_args$class == 2){
        if (LOCALIZATION=='FR')
          comment[4] <-  "Aspect de restriction d'hétérogénéité des gammaglobulines. Évolution à surveiller en fonction du contexte clinique."
        if (LOCALIZATION=='EN')
          comment[4] <-  "Restriction of immunoglobulin heterogeneity. Evolution to be monitored as appropriate to the clinical context."
      }

      if (input_args$g == -1 & input_args$class == 2){
        if (LOCALIZATION=='FR')
          comment[4] <-  "Aspect de restriction d'hétérogénéité des gammaglobulines dans un contexte d'hypogammaglobulinémie modérée. Évolution à surveiller en fonction du contexte clinique."
        if (LOCALIZATION=='EN')
          comment[4] <-  "Restriction of immunoglobulin heterogeneity with moderate hypogammaglobulinemia. Evolution to be monitored as appropriate to the clinical context."
      }

      if (input_args$g == -2 & input_args$class == 2){
        if (LOCALIZATION=='FR')
          comment[4] <-  "Aspect de restriction d'hétérogénéité des gammaglobulines dans un contexte d'hypogammaglobulinémie importante, évolution à surveiller. Identification immunologique recommandée en fonction du contexte clinique."
        if (LOCALIZATION=='EN')
          comment[4] <-  "Restriction of immunoglobulin heterogeneity with major hypogammaglobulinemia. Evolution to be monitored. Immunological identification is recommended as appropriate to the clinical context."
      }

      if (input_args$g == 1 & input_args$class == 2){
        if (LOCALIZATION=='FR')
          comment[4] <-  "Aspect de restriction d'hétérogénéité des gammaglobulines dans un contexte d'hypergammaglobulinémie modérée, évolution à surveiller. Identification immunologique recommandée en fonction du contexte clinique."
        if (LOCALIZATION=='EN')
          comment[4] <-  "Restriction of immunoglobulin heterogeneity with moderate hypergammaglobulinemia. Evolution to be monitored. Immunological identification is recommended as appropriate to the clinical context."
      }

      if (input_args$g == 2 & input_args$class == 2){
        if (LOCALIZATION=='FR')
          comment[4] <-  "Aspect de restriction d'hétérogénéité dans un contexte d'hypergammaglobulinémie importante, évolution à surveiller. Identification immunologique recommandée en fonction du contexte clinique."
        if (LOCALIZATION=='EN')
          comment[4] <-  "Restriction of immunoglobulin heterogeneity with major hypergammaglobulinemia. Evolution to be monitored. Immunological identification is recommended as appropriate to the clinical context."
      }
    }
  }

  # future comment about spike evolution
  # if (input_args$class == 3 & (!is.null(ant))){
  #   comment[3] <-  "Stabilité du pic monoclonal depuis l'électrophorèse précédente du jj/mm/aaaa."
  # }
  # if (input_args$class == 3 & (!is.null(ant))){
  #   comment[3] <-  "Diminution du pic monoclonal depuis l'électrophorèse précédente du jj/mm/aa d'environ XX %."
  # }
  #
  # if (input_args$class == 3 & (!is.null(ant))){
  #   comment[3] <-  "Augmentation du pic monoclonal depuis l'électrophorèse précédente du jj/mm/aa d'environ XX %."
  # }

  spikes_loc <- input_args$spikes
  spikes_loc <- spikes_loc[order(spikes_loc)]
  if (input_args$n_spikes > 0) {
    if (length(unique(spikes_loc)) > 1) {
      first_comment = TRUE
      tmp_comment <- ""
      for(i in 1:length(input_args$frac_names)) {
        n_spikes_in_frac <- sum(spikes_loc==i)
        if (n_spikes_in_frac > 0) {
          if (first_comment) {
            first_comment = FALSE
          } else {
            tmp_comment <- paste0(tmp_comment, ", ")
          }
          if (n_spikes_in_frac > 1) {
            if (LOCALIZATION=='FR')
              tmp_comment <- paste0(tmp_comment, n_spikes_in_frac, " pics d'aspect monoclonal en ", input_args$long_frac_names[i])
            if (LOCALIZATION=='EN')
              tmp_comment <- paste0(tmp_comment, n_spikes_in_frac, " suspected M-spikes in ", input_args$long_frac_names[i])
          } else {
            if (LOCALIZATION=='FR')
              tmp_comment <- paste0(tmp_comment, n_spikes_in_frac, " pic d'aspect monoclonal en ", input_args$long_frac_names[i])
            if (LOCALIZATION=='EN')
              tmp_comment <- paste0(tmp_comment, n_spikes_in_frac, " suspected M-spike in ", input_args$long_frac_names[i])
          }
        }
      }
      tmp_comment=paste0(tmp_comment,".")
      comment[1] = tmp_comment
    } else {
      if (input_args$n_spikes > 1) {
        if (LOCALIZATION=='FR')
          comment[1] <- paste0(input_args$n_spikes, " pics d'aspect monoclonal en ", input_args$long_frac_names[spikes_loc[1]], ".")
        if (LOCALIZATION=='EN')
          comment[1] <- paste0(input_args$n_spikes, " suspected M-spikes in ", input_args$long_frac_names[spikes_loc[1]], ".")
      } else {
        if (LOCALIZATION=='FR')
          comment[1] <- paste0("Pic d'aspect monoclonal en ", input_args$long_frac_names[spikes_loc[1]], ".")
        if (LOCALIZATION=='EN')
          comment[1] <- paste0("Suspected M-spike in ", input_args$long_frac_names[spikes_loc[1]], ".")
      }
    }
  }

  # spike comments
  # if (input_args$class == 5 & input_args$n_spikes == 0){
  #   comment[1] <-   "Augmentation importante des bêta-2 globulines. Évolution à surveiller en fonction du contexte clinique"
  # }

  if (input_args$class == 3 & !input_args$plasmapheresis & !input_args$nephrotic){
    if (is.null(input_args$gres)){
      if (input_args$g > 0) {
        if (LOCALIZATION=='FR')
          comment[2] <- "Augmentation des gammaglobulines."
        if (LOCALIZATION=='EN')
          comment[2] <- "Increase of gammaglobulins."
      } else if (input_args$g < 0) {
        if (LOCALIZATION=='FR')
          comment[2] <- "Diminution des gammaglobulines."
        if (LOCALIZATION=='EN')
          comment[2] <- "Decrease of gammaglobulins."
      } else {
        if (LOCALIZATION=='FR')
          comment[2] <- "Conservation des gammaglobulines."
        if (LOCALIZATION=='EN')
          comment[2] <- "Conservation of gammaglobulins."
      }
    }else{
      if (input_args$gres > 0) {
        if (LOCALIZATION=='FR')
          comment[2] <- "Augmentation des gammaglobulines résiduelles."
        if (LOCALIZATION=='EN')
          comment[2] <- "Increase of residual gammaglobulins."
      } else if (input_args$gres < 0) {
        if (LOCALIZATION=='FR')
          comment[2] <- "Diminution des gammaglobulines résiduelles."
        if (LOCALIZATION=='EN')
          comment[2] <- "Decrease of residual gammaglobulins."
      } else {
        if (LOCALIZATION=='FR')
          comment[2] <- "Conservation des gammaglobulines résiduelles."
        if (LOCALIZATION=='EN')
          comment[2] <- "Conservation of residual gammaglobulins."
      }
    }
  }

  # removing the NA to get a vector with our flag comments only
  comment <- comment[!is.na(comment)]

  printd('text_backend/commentMaker:out')

  return(list(text=comment, bin=qtyqly_normal))
}

# A RAJOUTER POUR MOSS 2016 : 
# beta-gamma bridging + aug beta
# increase/suppression gammaglobulins : + de stades

# version including suggestions from Moss (2016)
commentMaker_MOSS2016 <- function(input_args, ant=NULL){
  printd('text_backend/commentMaker:in')
  
  if (input_args$n_frac < 4 | input_args$n_frac > 6) {
    printd('text_backend/commentMaker:out(null)')
    if (LOCALIZATION=='FR') {
      return(list(text="Interprétation impossible.", bin=0))
    } else if (LOCALIZATION=='EN') {
      return(list(text="Interpretation impossible.", bin=0))
    }
  }
  
  # vector to store the flags comment
  comment <- c(rep(NA, 44))
  qtyqly_normal <- -1
  
  # Conditions for each comments
  if (input_args$prot == 0 & input_args$alb== 0 & input_args$albq== 0 & input_args$a1 == 0 & input_args$a2 == 0 & input_args$b1 == 0 
      & input_args$b2 == 0 & input_args$g == 0 & input_args$class == 1 & input_args$nephrotic == 0 & input_args$plasmapheresis == 0){
    qtyqly_normal <- 1
    if (LOCALIZATION=='FR')
      comment[12] <- "Profil qualitatif et quantitatif de l'électrophorèse sans anomalie notable."
    if (LOCALIZATION=='EN')
      comment[12] <- "Qualitative and quantitative electrophoresis profile without significant abnormalities."
  }

  # if (input_args$prot == 0 & input_args$alb== 0 & input_args$albq== 0 & input_args$a1 == 0 & input_args$a2 == 0 & input_args$b1 == 0 
  #     & input_args$b2 == 1 & input_args$g == 0 & input_args$class == 1 & input_args$nephrotic == 0 & input_args$plasmapheresis == 0){
  #   qtyqly_normal <- 1
  #   if (LOCALIZATION=='FR')
  #     comment[12] <- "Profil qualitatif et quantitatif de l'électrophorèse sans anomalie notable."
  #   if (LOCALIZATION=='EN')
  #     comment[12] <- "Qualitative and quantitative electrophoresis profile without significant abnormalities."
  # }
  
  # OK ----- MOSS
  if (input_args$nephrotic==1) {
    if (LOCALIZATION=='FR')
      comment[8] <- "Profil compatible avec un syndrome néphrotique. À confronter avec le contexte clinique et à confirmer par le dosage de la protéinurie des 24 heures."
    if (LOCALIZATION=='EN')
      comment[8] <- "Pattern is consistent with a nephrotic syndrome. Compare with the clinical context and 24-hour urine protein assay."
  }
  
  # ADDED ----- not initially in MOSS
  if (input_args$plasmapheresis==1){
    if (LOCALIZATION=='FR')
      comment[10] <- "Diminution globale des globulines : échange plasmatiques?"
    if (LOCALIZATION=='EN')
      comment[10] <- "Global decrease of globulins: plasmapheresis?"
  }
  
  # OK ----- MOSS
  if (input_args$hemolysis == 1){
    if (LOCALIZATION=='FR')
      comment[11] <- "Profil atypique compatible avec l'hémolyse observée sur l'échantillon. Il est recommandé de renvoyer un échantillon pour contrôle."
    if (LOCALIZATION=='EN')
      comment[11] <- "Atypical pattern is consistent with the hemolysis observed in this specimen. Suggest repeat on a fresh specimen."
  }
  
  # ADDED ----- not initially in MOSS
  global_hypoprot = F
  if (input_args$prot == 2 & input_args$alb < 1 & input_args$albq== 0 
      & input_args$a1 < 1 & input_args$a2 == 0 & input_args$b1 == 0 
      & input_args$b2 < 1 & input_args$g < 1
      & input_args$class == 1 & !input_args$plasmapheresis & !input_args$nephrotic){
    global_hypoprot = T
    if (LOCALIZATION=='FR')
      comment[10] <- "Hypoprotéinémie globale (fuite urinaire ou digestive, insuffisance hépatique, dénutrition, dilution)."
    if (LOCALIZATION=='EN')
      comment[10] <- "Global hypoproteinemia (urinary or digestive leakage, liver failure, undernutrition, dilution)."
  }
  
  # OK ----- MOSS
  if (!input_args$plasmapheresis & !input_args$nephrotic) {
    if (input_args$ped==1) {
      if (input_args$alb > -1 
          & (input_args$a1 == 1 | input_args$a2 > 0)){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil compatible avec un syndrome inflammatoire aigu/chronique."
        if (LOCALIZATION=='EN')
          comment[6] <- "Pattern is consistent with an acute/chronic inflammatory response."
      }
      
      if (input_args$alb == -1 
          & (input_args$a1 == 1 | input_args$a2 > 0 )){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil compatible avec un syndrome inflammatoire aigu/chronique et hypoalbuminémie."
        if (LOCALIZATION=='EN')
          comment[6] <- "Pattern is consistent with an acute/chronic inflammatory response with hypoalbuminemia."
      }
      
      if ((input_args$a1 == 1 | input_args$a2 > 0) & input_args$g == 1 & input_args$class==1){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil électrophorétique compatible avec un syndrome inflammatoire aigu/chronique accompagné d'une réaction immunitaire."
        if (LOCALIZATION=='EN')
          comment[6] <- "Pattern is consistent with an acute/chronic inflammatory response with an acute phase response."
      }
      
      if (input_args$alb == -1 & input_args$a1 == 0 & input_args$a2 == 0 & !global_hypoprot){
        if (LOCALIZATION=='FR')
          comment[7] <- "Hypoalbuminémie"
        if (LOCALIZATION=='EN')
          comment[7] <- "Hypoalbuminemia."
      }
    } else {
      if (input_args$alb > -1 
          & (input_args$a1 == 1 | input_args$a2 == 1)){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil compatible avec un syndrome inflammatoire aigu/chronique."
        if (LOCALIZATION=='EN')
          comment[6] <- "Pattern is consistent with an acute/chronic inflammatory response."
      }
      
      if (input_args$alb == -1 
          & (input_args$a1 == 1 | input_args$a2 == 1)){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil compatible avec un syndrome inflammatoire aigu/chronique et hypoalbuminémie."
        if (LOCALIZATION=='EN')
          comment[6] <- "Pattern is consistent with an acute/chronic inflammatory response with hypoalbuminemia."
      }
      
      if (input_args$alb == -2 
          & (input_args$a1 == 1 | input_args$a2 == 1)){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil compatible avec un syndrome inflammatoire aigu/chronique et hypoalbuminémie.."
        if (LOCALIZATION=='EN')
          comment[6] <- "Pattern is consistent with an acute/chronic inflammatory response with hypoalbuminemia."
      }
      
      if (input_args$alb > -1 
          & (input_args$a1 == 2 | input_args$a2 == 2)){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil compatible avec un syndrome inflammatoire aigu/chronique."
        if (LOCALIZATION=='EN')
          comment[6] <- "Pattern is consistent with an acute/chronic inflammatory response."
          # comment[6] <- "Profile compatible with a major inflammatory syndrome."
      }
      
      if (input_args$alb == -1 
          & (input_args$a1 == 2 | input_args$a2 == 2)){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil compatible avec un syndrome inflammatoire aigu/chronique et hypoalbuminémie.."
        if (LOCALIZATION=='EN')
          comment[6] <- "Pattern is consistent with an acute/chronic inflammatory response with hypoalbuminemia."
      }
      
      if (input_args$alb == -2 
          & (input_args$a1 == 2 | input_args$a2 == 2)){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil compatible avec un syndrome inflammatoire aigu/chronique et hypoalbuminémie."
        if (LOCALIZATION=='EN')
          comment[6] <- "Pattern is consistent with an acute/chronic inflammatory response with hypoalbuminemia."
      }
      
      if ((input_args$a1 > 0 | input_args$a2 > 0) & input_args$g > 0 & input_args$class==1){
        if (LOCALIZATION=='FR')
          comment[6] <- "Profil électrophorétique compatible avec un syndrome inflammatoire aigu/chronique accompagné d'une réaction immunitaire."
        if (LOCALIZATION=='EN')
          comment[6] <- "Pattern is consistent with an acute/chronic inflammatory response with an acute phase response."
      }
      
      if (input_args$alb == -1 & input_args$a1 == 0 & input_args$a2 == 0 & !global_hypoprot){
        if (LOCALIZATION=='FR')
          comment[7] <- "Hypoalbuminémie modérée."
        if (LOCALIZATION=='EN')
          comment[7] <- "Hypoalbuminemia."
      }
      
      if (input_args$alb == -2 & input_args$a1 == 0 & input_args$a2 == 0 & !global_hypoprot){
        if (LOCALIZATION=='FR')
          comment[7] <- "Hypoalbuminémie importante."
        if (LOCALIZATION=='EN')
          comment[7] <- "Hypoalbuminemia."
      }
    }
    
    if (input_args$prot == 0 & input_args$a1 == -1){
      if (LOCALIZATION=='FR')
        comment[9] <- "Diminution des alpha-1-globulines; il est recommandé d'effectuer un dosage et un phénotypage de l'alpha-1-antitrypsine, si indiqué."
      if (LOCALIZATION=='EN')
        comment[9] <- "Alpha-1 globulins decreased; consider alpha-1 antitrypsin quantitation and phenotyping, if indicated."
    }
  }
  
  # ADDED TO MOSS
  if (input_args$prot == 0 & input_args$alb== 1){
    if (LOCALIZATION=='FR')
      comment[7] <- "Augmentation de l'albumine, hémoconcentration très probable. À confronter au contexte clinique"
    if (LOCALIZATION=='EN')
      comment[7] <- "Increase of albumin, probable hemoconcentration. To be confronted with the clinical context."
  }
  
  # ADDED TO MOSS
  if (input_args$albq== 1){
    if (LOCALIZATION=='FR')
      comment[8] <- "Bisalbuminémie soit congénitale (variant de l'albumine) soit secondaire à un traitement par des bêtalactamines (pénicillines ou céphalosporines), ou une protéolyse par les enzymes pancréatiques libérées en excès. Évolution à surveiller si nécessaire."
    if (LOCALIZATION=='EN')
      comment[8] <- "Bisalbuminemia, either congenital (albumin variant) or secondary to a treatment by beta-lactamins or proteolysis by pancreatic enzymes. Evolution to be monitored."
  }
  
  # ADDED TO MOSS
  if (input_args$alb== -3){
    if (LOCALIZATION=='FR')
      comment[9] <-  "Absence d'albumine : analbuminémie congénitale avec augmentation de toute les fractions globuliniques."
    if (LOCALIZATION=='EN')
      comment[9] <-  "Absence of albumin: congenital analbuminemia with increase of all globulins."
  }
  
  # WITHDRAWN FOR MOSS
  if (FALSE) {
    # withdrawn because not in MOSS 2016
    if (input_args$age== 1 & input_args$b2b1 == 1 & input_args$g == -2 & input_args$class == 1 & !input_args$plasmapheresis & !input_args$nephrotic){
      if (LOCALIZATION=='FR')
        comment[5] <-  "Les bêta-2-globulines sont supérieures aux bêta-1-globulines avec diminution des gammaglobulines. L'identification immunologique est recommandée selon le contexte clinique."
      if (LOCALIZATION=='EN')
        comment[5] <-  "Beta-2-globulins are higher than beta-1-globulins, in a context of hypogammaglobluminemia. Immunological identification is recommended as appropriate to the clinical context."
    }
  }
  
  # OK ----- MOSS
  if (input_args$class == 4 & input_args$b1 <= 0 & input_args$b2 <= 0){
    if (LOCALIZATION=='FR')
      comment[4] <-  "Bloc bêta-gamma isolé."
    if (LOCALIZATION=='EN')
      comment[4] <- "Isolated beta-gamma bridging is noted."
  }
  
  # OK ----- MOSS
  if (input_args$class == 4 & (input_args$b1 > 0 | input_args$b2 > 0)){
    if (LOCALIZATION=='FR')
      comment[4] <-  "Bloc bêta-gamma avec augmentation des bêta-globulines."
    if (LOCALIZATION=='EN')
      comment[4] <- "Beta-gamma bridging is noted, with increased beta-globulins."
  }
  
  # OK ----- MOSS
  if (input_args$sex == 0 & input_args$b1 == 1 & input_args$class == 1){
    if (LOCALIZATION=='FR')
      comment[13] <-  "Augmentation des bêta-1-globulines (transferrine?)."
    if (LOCALIZATION=='EN')
      comment[13] <- "Relative increase in beta-1 globulins (transferrin?)."
  }
  
  # OK ----- MOSS
  if (input_args$sex == 1 & input_args$b1 == 1 & input_args$class == 1){
    if (LOCALIZATION=='FR')
      comment[13] <-  "Augmentation des bêta-1-globulines (transferrine?)."
    if (LOCALIZATION=='EN')
      comment[13] <- "Relative increase in beta-1 globulins (transferrin?)."
  }
  
  if (input_args$a1 <= 0 & input_args$a2 <= 0 & input_args$b2 == 1 & input_args$class == 1){
    if (LOCALIZATION=='FR')
      comment[5] <-   "Augmentation des bêta-2-globulines. Immunotypage ajouté."
    if (LOCALIZATION=='EN')
      comment[5] <-   "Relative increase in beta-2 globulins; IFE added."
  }
  
  # OK ----- MOSS
  # ne sera pas donné pour péd (b2>1)
  if (input_args$a1 <= 0 & input_args$a2 <= 0 & input_args$b2 == 2 & input_args$class == 1){
    if (LOCALIZATION=='FR')
      comment[5] <-   "Augmentation des bêta-2-globulines. Immunotypage ajouté."
    if (LOCALIZATION=='EN')
      comment[5] <-   "Relative increase in beta-2 globulins; IFE added."
  }
  
  # OK ----- MOSS
  if (input_args$b2 == -1 & !input_args$plasmapheresis & !input_args$nephrotic){
    if (LOCALIZATION=='FR')
      comment[14] <-   "Diminution importante des bêta-2-globulines; peut indiquer une diminution du complément ou une dégradation du prélèvement liée à la durée de stockage. Il est recommandé de réaliser un nouveau prélèvement."
    if (LOCALIZATION=='EN')
      comment[14] <- "Markedly decreased beta-2 globulins; may indicate decreased complement or degradation during prolonged storage. Consider repeat analysis on a fresh collection."
  }
  
  # TO PERFECT ----- MOSS
  if (!input_args$plasmapheresis & !input_args$nephrotic) {
    if (input_args$ped==1) {
      if (input_args$g == -1 & input_args$class == 1){
        if (LOCALIZATION=='FR')
          comment[4] <-   "Diminution des gammaglobulines"
        if (LOCALIZATION=='EN')
          comment[4] <-   "Suppression of gammaglobulins."
      }
      
      if (input_args$g == 1 & input_args$class == 1){
        if (LOCALIZATION=='FR')
          comment[4] <-   "Augmentation des gammaglobulines."
        if (LOCALIZATION=='EN')
          comment[4] <-   "Increase of gammaglobulins."
      }
      
      if (input_args$g == 0 & input_args$class == 2){
        if (LOCALIZATION=='FR')
          comment[4] <-  "Aspect de restriction d'hétérogénéité; évolution à surveiller."
        if (LOCALIZATION=='EN')
          comment[4] <-  "Restricted heterogeneity; follow-up is suggested."
      }
      
      if (input_args$g == -1 & input_args$class == 2){
        if (LOCALIZATION=='FR')
          comment[4] <-   "Aspect de restriction d'hétérogénéité avec diminution des gammaglobulines; évolution à surveiller."
        if (LOCALIZATION=='EN')
          comment[4] <-   "Restricted heterogeneity with suppression of gammaglobulins; follow-up is suggested."
      }
      
      if (input_args$g == 1 & input_args$class == 2){
        if (LOCALIZATION=='FR')
          comment[4] <-  "Aspect de restriction d'hétérogénéité avec augmentation des gammaglobulines; évolution à surveiller."
        if (LOCALIZATION=='EN')
          comment[4] <-  "Restricted heterogeneity with increase of gammaglobulins; follow-up is suggested."
      }
    } else {
      # MOSS 2016
      if (input_args$g < 0 & input_args$class == 1){
        if (input_args$g_prec == -1) {
          if (LOCALIZATION=='FR')
            comment[4] <-   "Faible diminution des gammaglobulines."
          if (LOCALIZATION=='EN')
            comment[4] <-   "Little gammaglobulins suppression."
        } else if (input_args$g_prec == -2) {
          if (LOCALIZATION=='FR')
            comment[4] <-   "Discrète diminution des gammaglobulines."
          if (LOCALIZATION=='EN')
            comment[4] <-   "Slight gammaglobulins suppression."
        } else if (input_args$g_prec == -3) {
          if (LOCALIZATION=='FR')
            comment[4] <-   "Diminution modérée des gammaglobulines."
          if (LOCALIZATION=='EN')
            comment[4] <-   "Moderate gammaglobulins suppression."
        } else if (input_args$g_prec == -4) {
          if (LOCALIZATION=='FR')
            comment[4] <-   "Diminution marquée des gammaglobulines."
          if (LOCALIZATION=='EN')
            comment[4] <-   "Marked gammaglobulins suppression."
        }
      }
      
      if (input_args$g > 0 & input_args$class == 1){
        if (input_args$g_prec == 1) {
          if (LOCALIZATION=='FR')
            comment[4] <-   "Faible augmentation des gammaglobulines."
          if (LOCALIZATION=='EN')
            comment[4] <-   "Little gammaglobulins increase."
        } else if (input_args$g_prec == 2) {
          if (LOCALIZATION=='FR')
            comment[4] <-   "Discrète augmentation des gammaglobulines."
          if (LOCALIZATION=='EN')
            comment[4] <-   "Slight gammaglobulins increase."
        } else if (input_args$g_prec == 3) {
          if (LOCALIZATION=='FR')
            comment[4] <-   "Augmentation modérée des gammaglobulines."
          if (LOCALIZATION=='EN')
            comment[4] <-   "Moderate gammaglobulins increase."
        } else if (input_args$g_prec == 4) {
          if (LOCALIZATION=='FR')
            comment[4] <-   "Augmentation marquée des gammaglobulines."
          if (LOCALIZATION=='EN')
            comment[4] <-   "Marked gammaglobulins increase."
        }
      }
      
      if (input_args$g < 0 & input_args$class == 2){
        if (input_args$g_prec == -1) {
          if (LOCALIZATION=='FR')
            comment[4] <-   "Restriction d'hétérogénéité avec diminution faible des gammaglobulines; évolution à surveiller."
          if (LOCALIZATION=='EN')
            comment[4] <-   "Restricted heterogeneity with little gammaglobulins suppression; follow-up is suggested."
        } else if (input_args$g_prec == -2) {
          if (LOCALIZATION=='FR')
            comment[4] <-   "Restriction d'hétérogénéité avec discrète diminution des gammaglobulines; évolution à surveiller."
          if (LOCALIZATION=='EN')
            comment[4] <-   "Restricted heterogeneity with slight gammaglobulins suppression; follow-up is suggested."
        } else if (input_args$g_prec == -3) {
          if (LOCALIZATION=='FR')
            comment[4] <-   "Restriction d'hétérogénéité avec diminution modérée des gammaglobulines; évolution à surveiller."
          if (LOCALIZATION=='EN')
            comment[4] <-   "Restricted heterogeneity with moderate gammaglobulins suppression; follow-up is suggested."
        } else if (input_args$g_prec == -4) {
          if (LOCALIZATION=='FR')
            comment[4] <-   "Restriction d'hétérogénéité avec diminution marquée des gammaglobulines; évolution à surveiller."
          if (LOCALIZATION=='EN')
            comment[4] <-   "Restricted heterogeneity with marked gammaglobulins suppression; follow-up is suggested."
        }
      }
      
      if (input_args$g > 0 & input_args$class == 2){
        if (input_args$g_prec == 1) {
          if (LOCALIZATION=='FR')
            comment[4] <-   "Restriction d'hétérogénéité avec augmentation faible des gammaglobulines; évolution à surveiller."
          if (LOCALIZATION=='EN')
            comment[4] <-   "Restricted heterogeneity with little gammaglobulins increase; follow-up is suggested."
        } else if (input_args$g_prec == 2) {
          if (LOCALIZATION=='FR')
            comment[4] <-   "Restriction d'hétérogénéité avec discrète augmentation des gammaglobulines; évolution à surveiller."
          if (LOCALIZATION=='EN')
            comment[4] <-   "Restricted heterogeneity with slight gammaglobulins increase; follow-up is suggested."
        } else if (input_args$g_prec == 3) {
          if (LOCALIZATION=='FR')
            comment[4] <-   "Restriction d'hétérogénéité avec augmentation modérée des gammaglobulines; évolution à surveiller."
          if (LOCALIZATION=='EN')
            comment[4] <-   "Restricted heterogeneity with moderate gammaglobulins increase; follow-up is suggested."
        } else if (input_args$g_prec == 4) {
          if (LOCALIZATION=='FR')
            comment[4] <-   "Restriction d'hétérogénéité avec augmentation marquée des gammaglobulines; évolution à surveiller."
          if (LOCALIZATION=='EN')
            comment[4] <-   "Restricted heterogeneity with marked gammaglobulins increase; follow-up is suggested."
        }
      }
    }
  }
  
  # OK MOSS
  spikes_loc <- input_args$spikes
  spikes_loc <- spikes_loc[order(spikes_loc)]
  if (input_args$n_spikes > 0) {
    if (length(unique(spikes_loc)) > 1) {
      first_comment = TRUE
      tmp_comment <- ""
      for(i in 1:length(input_args$frac_names)) {
        n_spikes_in_frac <- sum(spikes_loc==i)
        if (n_spikes_in_frac > 0) {
          if (first_comment) {
            first_comment = FALSE
          } else {
            tmp_comment <- paste0(tmp_comment, ", ")
          }
          if (n_spikes_in_frac > 1) {
            if (LOCALIZATION=='FR')
              tmp_comment <- paste0(tmp_comment, n_spikes_in_frac, " pics d'aspect monoclonal en ", input_args$long_frac_names[i])
            if (LOCALIZATION=='EN')
              tmp_comment <- paste0(tmp_comment, n_spikes_in_frac, " new apparent paraproteins in ", input_args$long_frac_names[i])
          } else {
            if (LOCALIZATION=='FR')
              tmp_comment <- paste0(tmp_comment, n_spikes_in_frac, " pic d'aspect monoclonal en ", input_args$long_frac_names[i])
            if (LOCALIZATION=='EN')
              tmp_comment <- paste0(tmp_comment, n_spikes_in_frac, " new apparent paraprotein in ", input_args$long_frac_names[i])
          }
        }
      }
      tmp_comment=paste0(tmp_comment,".")
      comment[1] = tmp_comment
    } else {
      if (input_args$n_spikes > 1) {
        if (LOCALIZATION=='FR')
          comment[1] <- paste0(input_args$n_spikes, " pics d'aspect monoclonal en ", input_args$long_frac_names[spikes_loc[1]], ".")
        if (LOCALIZATION=='EN')
          comment[1] <- paste0(input_args$n_spikes, " new apparent paraproteins in ", input_args$long_frac_names[spikes_loc[1]], ".")
      } else {
        if (LOCALIZATION=='FR')
          comment[1] <- paste0("Pic d'aspect monoclonal en ", input_args$long_frac_names[spikes_loc[1]], ".")
        if (LOCALIZATION=='EN')
          comment[1] <- paste0("New apparent paraprotein in ", input_args$long_frac_names[spikes_loc[1]], ".")
      }
    }
  }
  
  # spike comments
  # if (input_args$class == 5 & input_args$n_spikes == 0){
  #   comment[1] <-   "Augmentation importante des bêta-2 globulines. Évolution à surveiller en fonction du contexte clinique"
  # }
  
  if (input_args$class == 3 & !input_args$plasmapheresis & !input_args$nephrotic){
    if (input_args$ped==1) {
      if (is.null(input_args$gres)){
        if (input_args$g > 0) {
          if (LOCALIZATION=='FR')
            comment[2] <- "Augmentation des gammaglobulines."
          if (LOCALIZATION=='EN')
            comment[2] <- "Increase of gammaglobulins."
        } else if (input_args$g < 0) {
          if (LOCALIZATION=='FR')
            comment[2] <- "Diminution des gammaglobulines."
          if (LOCALIZATION=='EN')
            comment[2] <- "Suppression of gammaglobulins."
        } else {
          if (LOCALIZATION=='FR')
            comment[2] <- "Conservation des gammaglobulines."
          if (LOCALIZATION=='EN')
            comment[2] <- "No apparent suppression of gammaglobulins."
        }
      }else{
        if (input_args$gres > 0) {
          if (LOCALIZATION=='FR')
            comment[2] <- "Augmentation des gammaglobulines résiduelles."
          if (LOCALIZATION=='EN')
            comment[2] <- "Increase of residual gammaglobulins."
        } else if (input_args$gres < 0) {
          if (LOCALIZATION=='FR')
            comment[2] <- "Diminution des gammaglobulines résiduelles."
          if (LOCALIZATION=='EN')
            comment[2] <- "Suppression of residual gammaglobulins."
        } else {
          if (LOCALIZATION=='FR')
            comment[2] <- "Conservation des gammaglobulines résiduelles."
          if (LOCALIZATION=='EN')
            comment[2] <- "No apparent suppression of residual gammaglobulins."
        }
      }
      
    } else {
      if (is.null(input_args$gres)){
        g_value <- input_args$g_prec
        if (LOCALIZATION=='FR')
          g_name <- "gammaglobulines"
        if (LOCALIZATION=='EN')
          g_name <- "gammaglobulins"
      } else {
        g_value <- input_args$gres_prec
        if (LOCALIZATION=='FR')
          g_name <- "gammaglobulines résiduelles"
        if (LOCALIZATION=='EN')
          g_name <- "residual gammaglobulins"
      }
      if (g_value == 1) {
        if (LOCALIZATION=='FR')
          comment[2] <- paste("Faible augmentation des",g_name,".")
        if (LOCALIZATION=='EN')
          comment[2] <- paste("Little",g_name,"increase.")
      } else if (g_value == 2) {
        if (LOCALIZATION=='FR')
          comment[2] <- paste("Discrète augmentation des",g_name,".")
        if (LOCALIZATION=='EN')
          comment[2] <- paste("Slight",g_name,"increase.")
      } else if (g_value == 3) {
        if (LOCALIZATION=='FR')
          comment[2] <- paste("Augmentation modérée des",g_name,".")
        if (LOCALIZATION=='EN')
          comment[2] <- paste("Moderate",g_name,"increase.")
      } else if (g_value == 4) {
        if (LOCALIZATION=='FR')
          comment[2] <- paste("Augmentation importante des",g_name,".")
        if (LOCALIZATION=='EN')
          comment[2] <- paste("Marked",g_name,"increase.")
      } else if (g_value == -1) {
        if (LOCALIZATION=='FR')
          comment[2] <- paste("Faible diminution des",g_name,".")
        if (LOCALIZATION=='EN')
          comment[2] <- paste("Little",g_name,"suppression.")
      } else if (g_value == -2) {
        if (LOCALIZATION=='FR')
          comment[2] <- paste("Discrète diminution des",g_name,".")
        if (LOCALIZATION=='EN')
          comment[2] <- paste("Slight",g_name,"suppression.")
      } else if (g_value == -3) {
        if (LOCALIZATION=='FR')
          comment[2] <- paste("Diminution modérée des",g_name,".")
        if (LOCALIZATION=='EN')
          comment[2] <- paste("Moderate",g_name,"suppression.")
      } else if (g_value == -4) {
        if (LOCALIZATION=='FR')
          comment[2] <- paste("Diminution importante des",g_name,".")
        if (LOCALIZATION=='EN')
          comment[2] <- paste("Marked",g_name,"suppression.")
      } else {
        if (LOCALIZATION=='FR')
          comment[2] <- "Conservation des gammaglobulines."
        if (LOCALIZATION=='EN')
          comment[2] <- "No apparent suppression of gammaglobulins."
      }
    }
  }
  
  # removing the NA to get a vector with our flag comments only
  comment <- comment[!is.na(comment)]
  
  printd('text_backend/commentMaker:out')
  
  return(list(text=comment, bin=qtyqly_normal))
}

RECOMMENDATIONS = list(list(name="Szymanowicz et al., 2006", fn=commentMaker_SZYM2006, shortname="SZYM"),
                       list(name="Moss MA, 2016", fn=commentMaker_MOSS2016, shortname="MOSS"))

##################################################################################################
########################################## SERVER ################################################
##################################################################################################

last_input <- as.numeric(Sys.time())*1000
last_refresh <- as.numeric(Sys.time())*1000

usedRecommendations <- RECOMMENDATIONS[[1]]

if (QUICK_MINIATURES) {
  thumbnail_template <- image_read(path=thumbnail_template_path)
}

server=function(input, output, session) {
  # Create a buffer to store data when loaded or modified
  buffered_data <- reactiveValues()
  # buffered_data <- list()
  buffered_data$batch_data <- NULL
  buffered_data$current_sample <- 0
  buffered_data$cursor_status <- 0
  buffered_data$spike_start <- 0
  
  disclaimer_data <- reactiveValues()
  disclaimer_data$validated = FALSE
  
  output$disclaimer_validated <- reactive({
    return(disclaimer_data$validated)
  })
  
  # last button
  observeEvent(input$disclaimer_valid, {
    printd('validated disclaimer', level=1)
    disclaimer_data$validated <- TRUE
  })

  # JS argument: used to know if there's a batch loaded right now
  output$currently_active_batch <- reactive({
    printd('server/currently_active_batch')
    return(!is.null(buffered_data$batch_data))
  })

  # used to know if we're in debug mode
  output$debug_mode <- reactive({
    printd(paste0('Debug mode: ',DEBUG_MODE))
    return(DEBUG_MODE)
  })

  # JS argument: used to know which tab we're viewing
  output$batch_sample_active <- reactive({
    printd('server/batch_sample_active')
    return(input$main_tabs=='tab_sample')
  })
  
  # Output "manual comment" or "automatic comment" according to user choice
  output$comment_title <- renderUI({
    printd('server/comment_title:in')
    
    if (is.null(buffered_data$batch_data)) {
      printd('server/comment_title:out(null)')
      return(NULL)
    }
    
    tmp <- buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]
    
    printd('server/comment_title:out')
    HTML('Automatic interpretation:')
  })
  
  loadJSONBatch <- function(json_filepath) {
    printd('server/loadJSONBatch:in')
    
    if (!is.null(buffered_data$batch_data)) {
      printd('server/loadJSONBatch:out(alreadyloaded')
      return()
    }
    
    if (USE_SHINYJS) {
      disable('load_demo_batch')
      disable('load_expert_batch')
      disable('load_expert_batch2')
      disable('auto_analyze_input')
      disable('single_upload')
      disable('batch_upload')
      disable('img_file_upload')
      disable('img_file_modal')
      disable('single_file_text_ok')
      disable('download_json_template')
      disable('download_img_template')
      disable('download_img_tutorial')
      disable('reload_results')
      disable('save_button')
    }

    buffered_data$batch_data <- NULL
    buffered_data$current_sample <- 0
    buffered_data$cursor_status <- 0
    buffered_data$spike_start <- 0
    buffered_data$miniatures <- NULL
    
    tmp <- fromJSON(paste(readLines(json_filepath), collapse=""))
    # we get the number of samples
    total_n=length(tmp$batch$samples)
    
    withProgress(message = 'Analyzing batch...', value = 0, {
      withProgress(message = 'Reading data...', value = NULL, {
        # extract batch data for cnn analysis
        # batch_data = t(matrix(unlist(sapply(tmp$batch$samples, function(a) a$original_curve_y)), ncol = length(tmp$batch$samples), nrow = 300))
        # now we also have to load metadata
        batch_metadata = matrix(c(sapply(tmp$batch$samples, function(a) a$age),
                                  sapply(tmp$batch$samples, function(a) a$total_protein)), nrow = length(tmp$batch$samples), ncol = 2)
      })
      # call cnn analysis
      incProgress(.33)
      if (input$auto_analyze_input) {
        withProgress(message = 'Running CNN analysis...', value = NULL, {
          # 1. output data R -> py
          tmp_py_json = list()
          tmp_py_json$id = tmp$batch$batch_id
          tmp_py_json$samples = list()
          for(i in 1:total_n) {
            tmp_py_json$samples[[i]] = list()
            tmp_py_json$samples[[i]]$id = tmp$batch$samples[[i]]$sample_id
            tmp_py_json$samples[[i]]$data = tmp$batch$samples[[i]]$original_curve_y
          }
          temp_id = paste0(session$request$REMOTE_ADDR,'-',round(as.numeric(Sys.time())*1000))
          temp_fileout_path = paste0(py_temp_path,"/",temp_id,".txt")
          temp_filein_path = paste0(py_temp_path,"/",temp_id,"r.txt")
          f <- file(temp_fileout_path)
          writeLines(toJSON(tmp_py_json), f)
          close(f)
          # 2. launch python script
          if (RETICULATE) {
            incProgress(0, detail = "Loading python...")
            if (is.null(ai)) {
              ai <<- import(gsub(".py","",py_script_path)) # import python functions
            }
            incProgress(.33, detail = "Loading models...")
            if (TWOSTEP_ANALYSIS) { # load models, then run analysis
              if (is.null(loaded_models)) {
                loaded_models <<- ai$LOADMODELS(model_path = paste0(getwd(), '/', py_models_path),
                                                save_format = 'tf')
              }
            }
            incProgress(.33, detail = "Inference...")
            tic=as.numeric(Sys.time())
            ai_out <- ai$SPECTR(data = paste0(getwd(), '/', temp_fileout_path),
                                output = paste0(getwd(), '/', temp_filein_path),
                                model_path = paste0(getwd(), '/', py_models_path),
                                loaded_models = loaded_models)
            toc=as.numeric(Sys.time())
          } else {
            tic=as.numeric(Sys.time())
            system_cmd <- paste0(python_path, ' ', py_script_path, ' --data "', getwd(), '/', temp_fileout_path, '" --output "', getwd(), '/', temp_filein_path, '" --model_path "', getwd(), '/', py_models_path, '"')
            printd(paste0("Launching system command: '", system_cmd, "'"))
            system(system_cmd)
            toc=as.numeric(Sys.time())
          }
          # 3. get results py -> R
          cnn_results <- fromJSON(paste(readLines(temp_filein_path), collapse=""))
          predictions <- R_analyze_raw_predictions(cnn_results,
                                                   batch_metadata,
                                                   FALSE)
        })
        elapsed_sec=toc-tic
        min_char=as.character(floor(elapsed_sec/60))
        sec_char=as.character(round(elapsed_sec%%60))
        if (nchar(sec_char)<2)
          sec_char=paste0('0',sec_char)
        time_char=paste0(min_char,'m',sec_char,'s')
        showNotification(paste0("CNN analysis ended in ",time_char," (",format(elapsed_sec/total_n, digits=2, nsmall=2),"s per sample)"))
      }
      
      analysis_msg <- "Reloading samples data..."
      if (input$auto_analyze_input) {
        analysis_msg <- "Analyzing CNN output..."
      }
      
      incProgress(.33)
      withProgress(message = analysis_msg, value = 0, {
        if (!input$auto_analyze_input) {
          predictions='ok'
        }
        if (!is.null(predictions)) {
          # set analysis data
          for(i in 1:total_n) {
            incProgress(1/total_n, detail=paste0(i, '/', total_n))
            # store analysis results for fractioning
            if (input$auto_analyze_input) {
              # standardize curve
              tmp$batch$samples[[i]]$original_curve_y = cnn_results[[i]]$x
              f <- predictions[[1]][i,]
              # Prevent possible (though unlikely) errors
              f <- f[order(f)]
              f[f > 303] <- 303
              f[f < 2] <- 2
              # check fraction boundaries as well as bisalbuminemia
              adjust_flag <- checkSegmentation(tmp$batch$samples[[i]]$original_curve_y, f)
              bisalb_flag <- seekBisalbuminemia(tmp$batch$samples[[i]]$original_curve_y, f)
              # and classification result
              top_class = which.max(predictions[[2]][i,])
              tmp$batch$samples[[i]]$computer_analysis <- list()
              tmp$batch$samples[[i]]$computer_analysis$predicted_class = top_class
              tmp$batch$samples[[i]]$computer_analysis$predicted_class_confidence <- predictions[[2]][i,top_class]
              tmp$batch$samples[[i]]$computer_analysis$predicted_class_confidences = as.numeric(predictions[[2]][i,])
              tmp$batch$samples[[i]]$computer_analysis$predicted_haemolysis = predictions[[4]][i,2]
              tmp$batch$samples[[i]]$computer_analysis$predicted_haemolysis_score = predictions[[4]][i,1]
              tmp$batch$samples[[i]]$computer_analysis$predicted_fractions_uncertainty=predictions[[5]][i]
              tmp$batch$samples[[i]]$computer_analysis$predicted_fractions_confidence=predictions[[6]][i]
              tmp$batch$samples[[i]]$computer_analysis$predmap_f1=predictions[[7]][i,]
              tmp$batch$samples[[i]]$computer_analysis$predmap_f2=predictions[[8]][i,]
              tmp$batch$samples[[i]]$computer_analysis$predmap_s1=predictions[[9]][i,]
              # store flags
              tmp$batch$samples[[i]]$computer_analysis$adjustment_flag <- adjust_flag
              tmp$batch$samples[[i]]$computer_analysis$bisalbuminemia_flag <- bisalb_flag
              tmp$batch$samples[[i]]$class <- top_class # default = predicted
              # finally, process spikes results, only if class == clonal anomaly
              if (top_class == 3) {
                spikes_vec <- predictions[[3]][i,]
                spikes_vec <- spikes_vec[spikes_vec!= 0]
                spikes_n <- length(spikes_vec)/3
              } else {
                spikes_n <- 0
              }
              s <- data.frame(index = numeric(spikes_n),
                              confidence = numeric(spikes_n),
                              start = numeric(spikes_n),
                              end = numeric(spikes_n),
                              qty_pct = numeric(spikes_n),
                              qty_abs = numeric(spikes_n),
                              loc = numeric(spikes_n),
                              stringsAsFactors = F)
              if (spikes_n > 0) {
                s$index = 1:spikes_n
                s$confidence = spikes_vec[seq(1, spikes_n*3, 3)]
                s$start = spikes_vec[seq(2, spikes_n*3, 3)]
                s$end = spikes_vec[seq(3, spikes_n*3, 3)]
                # reorder spikes by increasing start
                s=s[order(s$start),]
              }
              # store data
              tmp$batch$samples[[i]]$boundaries = f
              tmp$batch$samples[[i]]$spikes = s
              tmp$batch$samples[[i]]$computer_analysis$predicted_boundaries = f
              tmp$batch$samples[[i]]$computer_analysis$predicted_spikes = s
              tmp$batch$samples[[i]]$haemolysis = 0
              tmp$batch$samples[[i]]$locked = 0
              tmp$batch$samples[[i]]$bin_analysis = 0
            } else {
              # standardize curve
              tmp$batch$samples[[i]]$original_curve_y <- tmp$batch$samples[[i]]$original_curve_y/max(tmp$batch$samples[[i]]$original_curve_y)
              # and add zero padding
              spe_width=304
              total_pad = spe_width-length(tmp$batch$samples[[i]]$original_curve_y)
              start_pad = floor(total_pad/2)
              end_pad = total_pad-start_pad
              tmp$batch$samples[[i]]$original_curve_y <- c(rep(0,start_pad),tmp$batch$samples[[i]]$original_curve_y,rep(0,end_pad))

              tmp$batch$samples[[i]]$computer_analysis <- list()
              # add fake classification result
              tmp$batch$samples[[i]]$computer_analysis$predicted_class = 1
              tmp$batch$samples[[i]]$computer_analysis$predicted_class_confidence <- 1.
              tmp$batch$samples[[i]]$computer_analysis$predicted_class_confidences = c(1.,0.,0.,0.)
              tmp$batch$samples[[i]]$computer_analysis$predicted_haemolysis = 0
              tmp$batch$samples[[i]]$computer_analysis$predicted_haemolysis_score = 0.
              tmp$batch$samples[[i]]$computer_analysis$predicted_fractions_uncertainty = 0
              tmp$batch$samples[[i]]$computer_analysis$predicted_fractions_confidence = 1
              tmp$batch$samples[[i]]$computer_analysis$predmap_f1 = rep(0, spe_width)
              tmp$batch$samples[[i]]$computer_analysis$predmap_f2 = rep(1., spe_width)
              tmp$batch$samples[[i]]$computer_analysis$predmap_s1 = rep(0., spe_width)
              # store flags
              tmp$batch$samples[[i]]$computer_analysis$adjustment_flag <- FALSE
              tmp$batch$samples[[i]]$computer_analysis$bisalbuminemia_flag <- 0
              tmp$batch$samples[[i]]$class <- 1
              # store data
              tmp$batch$samples[[i]]$boundaries = numeric(0)
              tmp$batch$samples[[i]]$spikes = setNames(data.frame(matrix(ncol = 7, nrow = 0)), c('index','confidence','start','end','qty_pct','qty_abs','loc'))
              tmp$batch$samples[[i]]$computer_analysis$predicted_boundaries = numeric(0)
              tmp$batch$samples[[i]]$computer_analysis$predicted_spikes = setNames(data.frame(matrix(ncol = 7, nrow = 0)), c('index','confidence','start','end','qty_pct','qty_abs','loc'))
              tmp$batch$samples[[i]]$haemolysis = 0
              tmp$batch$samples[[i]]$locked = 0
              tmp$batch$samples[[i]]$bin_analysis = 0
            }
            # if (DEBUG_MODE & i==1) {
            #   printd(tmp$batch$samples[[i]])
            # }
            # compute first analysis
            tmp$batch$samples[[i]]=sampleAnalysis(tmp$batch$samples[[i]], 0, 0, i, usedRecommendations, DEBUG_MODE, FALSE) # without precomputing plots
          }
          # store data in buffer
          buffered_data$batch_data <- tmp
          buffered_data$current_sample <- 1
          # recall backend functions
          # refreshAnalysis()
          # update indexer and disable invalidate button
          # disable(id = 'invalid_button')
          # updateNumericInput(session, "sample_index_input", value = 1,
          #                    min = 1, max = length(tmp$batch$samples), step = 1)
          refreshInputUI()
        } else {
          showNotification(paste0("Unable to find models' predictions"), type="error")
        }
      })
    })
    if (USE_SHINYJS) {
      enable('load_demo_batch')
      enable('load_expert_batch')
      enable('load_expert_batch2')
      enable('auto_analyze_input')
      enable('single_upload')
      enable('batch_upload')
      enable('img_file_upload')
      enable('img_file_modal')
      enable('single_file_text_ok')
      enable('download_json_template')
      enable('download_img_template')
      enable('download_img_tutorial')
      enable('reload_results')
      enable('save_button')
    }
    
    if (!is.null(buffered_data$batch_data)) {
      updateTabsetPanel(session, "main_tabs", selected = "tab_sample")
    }
    printd('server/loadJSONBatch:out')
  }
  
  observeEvent(input$recom_input, {
    for(i in 1:length(RECOMMENDATIONS)) {
      if (input$recom_input == RECOMMENDATIONS[[i]]$name) {
        usedRecommendations <<- RECOMMENDATIONS[[i]]
        printd('new reco:', level=1)
        printd(RECOMMENDATIONS[[i]]$name, level=1)
        # printd(usedRecommendations, level=1)
        break
      }
    }
  })
  
  # When we click on "Load demo batch"
  observeEvent(input$load_demo_batch, {
    printd('server/load_demo_batch:in')
    
    now <- as.numeric(Sys.time())*1000
    if (now-last_input>DELAY_BETWEEN_INPUTS) {
      last_input <<- now
    } else {
      printd('server/load_demo_batch:out(toosoon)')
      return()
    }
    
    loadJSONBatch(demo_batch_path)
    printd('server/load_demo_batch:out')
  })
  
  # When we click on "Load expert batch"
  observeEvent(input$load_expert_batch, {
    printd('server/load_expert_batch:in')
    
    now <- as.numeric(Sys.time())*1000
    if (now-last_input>DELAY_BETWEEN_INPUTS) {
      last_input <<- now
    } else {
      printd('server/load_expert_batch:out(toosoon)')
      return()
    }
    
    loadJSONBatch(expert_batch_path)
    printd('server/load_expert_batch:out')
  })
  observeEvent(input$load_expert_batch2, {
    printd('server/load_expert_batch2:in')
    
    now <- as.numeric(Sys.time())*1000
    if (now-last_input>DELAY_BETWEEN_INPUTS) {
      last_input <<- now
    } else {
      printd('server/load_expert_batch2:out(toosoon)')
      return()
    }
    
    loadJSONBatch(expert_batch_path2)
    printd('server/load_expert_batch2:out')
  })

  # When we click on "Load Image"
  observeEvent(input$img_file_upload, {
    printd('server/img_file_upload:in')
    
    toggleModal(session, "img_file_modal", toggle = "close")
    
    now <- as.numeric(Sys.time())*1000
    if (now-last_input>DELAY_BETWEEN_INPUTS) {
      last_input <<- now
    } else {
      printd('server/img_file_upload:out(toosoon)')
      return()
    }
    
    if (!is.null(buffered_data$batch_data)) {
      printd('server/img_file_upload:out(alreadyloaded')
      return()
    }
    
    theFile <- input$img_file_upload
    if(!is.null(theFile)) {
      # printd(paste0('Loading image file: <', names(theFile), '>'), level=1)
      # load img file
      sample_id <- theFile$name
      age <- input$img_file_age
      sex <- input$img_file_sex
      total_protein <- input$img_file_tp
      raw_buffer=NULL
      try({raw_buffer=image_read(theFile$datapath)})
      # qc check
      qc_check_ok = T
      if (is.null(raw_buffer)) {
        showNotification(paste0("Image data not valid"), type="error")
        qc_check_ok = F
      } else if (image_info(raw_buffer)$height<100 | image_info(raw_buffer)$width<300) {
        showNotification(paste0("Image resolution is too low, expected width >= 300px and height >= 200px"), type="error")
        qc_check_ok = F
      }
      if (!qc_check_ok) {
        printd('server/csf_file:out(inconsistent)')
        showNotification(paste0("Unable to load image file"), type="error")
        return()
      }
      y <- IMG2SPECTR(raw_buffer)
      # qc check 2
      if (length(sex)==0) {
        showNotification(paste0("Please fill in the field: 'sex'"), type="error")
        qc_check_ok = F
      } else if (sex!="F" & sex!="M") {
        showNotification(paste0("Sex should be either 'M' or 'F'"), type="error")
        qc_check_ok = F
      }
      if (length(age)==0) {
        showNotification(paste0("Please fill in the field: 'age'"), type="error")
        qc_check_ok = F
      } else if (is.na(age)) {
        showNotification(paste0("Invalid age, please check that age is a numeric value"), type="error")
        qc_check_ok = F
      }
      if (length(total_protein)==0) {
        showNotification(paste0("Please fill in the field: 'total protein'"), type="error")
        qc_check_ok = F
      } else if (is.na(total_protein)) {
        showNotification(paste0("Invalid total protein, please check that total protein is a numeric value"), type="error")
        qc_check_ok = F
      }
      if (sum(is.na(y))>0) {
        showNotification(paste0("Invalid values in SPE curve, please check file structure"), type="error")
        qc_check_ok = F
      }
      if (length(y)!=300) {
        showNotification(paste0("Unable to obtain a valid 300 points curve, please check the image data"), type="error")
        qc_check_ok = F
      }
      if (!qc_check_ok) {
        printd('server/csf_file:out(inconsistent)')
        showNotification(paste0("Unable to load image file"), type="error")
        return()
      }
      # create json data file
      temp_id = paste0(session$request$REMOTE_ADDR,'-',round(as.numeric(Sys.time())*1000))
      temp_json_path = paste0(py_temp_path,"/",temp_id,"_temp.json")
      batch_full_id=temp_id
      json <- paste0('{\n\t"batch": {\n\t\t"batch_id": "', batch_full_id, '",\n\t\t"samples": [')
      json <- paste0(json, "\n\t\t{")
      json <- paste0(json, '\n\t\t\t"sample_id": "', sample_id,'",')
      json <- paste0(json, '\n\t\t\t"sex": "', sex,'",')
      json <- paste0(json, '\n\t\t\t"age": ', age, ',')
      json <- paste0(json, '\n\t\t\t"total_protein": ', total_protein, ',')
      json <- paste0(json, '\n\t\t\t"original_curve_y": [', paste(y, collapse = ','), ']')
      json <- paste0(json, "\n\t\t}")
      json <- paste0(json, ']\n\t}\n}')
      write(json, temp_json_path)
      # load json file
      loadJSONBatch(temp_json_path)
    }
    printd('server/img_file_upload:out')
  })
  
  # load json file
  observeEvent(input$batch_upload, {
    printd('server/batch_upload:in')
    
    now <- as.numeric(Sys.time())*1000
    if (now-last_input>DELAY_BETWEEN_INPUTS) {
      last_input <<- now
    } else {
      printd('server/batch_upload:out(toosoon)')
      return()
    }
    
    if (!is.null(buffered_data$batch_data)) {
      printd('server/batch_upload:out(alreadyloaded')
      return()
    }
    
    theFile <- input$batch_upload
    
    if(!is.null(theFile)) {
      if (!theFile$type=="application/json") {
        printd('server/batch_upload:out(inconsistent)')
        showNotification(paste0("Please select a valid json file"), type="error")
        return()
      }

      # load json file
      loadJSONBatch(theFile$datapath)
    }
    
    printd('server/batch_upload:out')
  })
  
  # upload sample as DI
  observeEvent(input$single_file_text_ok, {
    printd('server/single_file_text_ok:in')
    
    toggleModal(session, "img_file_modal", toggle = "close")
    
    now <- as.numeric(Sys.time())*1000
    if (now-last_input>DELAY_BETWEEN_INPUTS) {
      last_input <<- now
    } else {
      printd('server/single_file_text_ok:out(toosoon)')
      return()
    }
    
    if (!is.null(buffered_data$batch_data)) {
      printd('server/single_file_text_ok:out(alreadyloaded')
      return()
    }
    
    theText <- input$single_file_text
    if(!is.null(theText) & nchar(theText)>0) {
      # load img file
      sample_id <- "TextInput"
      age <- input$img_file_age
      sex <- input$img_file_sex
      total_protein <- input$img_file_tp
      # qc check
      qc_check_ok = T
      if (!grepl("^[0-9,]+$", theText)) {
        showNotification(paste0("Text data must contain only numbers and commas (,)"), type="error")
        qc_check_ok = F
      }
      if (!qc_check_ok) {
        printd('server/single_file_text:out(inconsistent)')
        showNotification(paste0("Unable to load image file"), type="error")
        return()
      }
      y <- as.numeric(strsplit(theText, ",")[[1]])
      # qc check 2
      if (length(sex)==0) {
        showNotification(paste0("Please fill in the field: 'sex'"), type="error")
        qc_check_ok = F
      } else if (sex!="F" & sex!="M") {
        showNotification(paste0("Sex should be either 'M' or 'F'"), type="error")
        qc_check_ok = F
      }
      if (length(age)==0) {
        showNotification(paste0("Please fill in the field: 'age'"), type="error")
        qc_check_ok = F
      } else if (is.na(age)) {
        showNotification(paste0("Invalid age, please check that age is a numeric value"), type="error")
        qc_check_ok = F
      }
      if (length(total_protein)==0) {
        showNotification(paste0("Please fill in the field: 'total protein'"), type="error")
        qc_check_ok = F
      } else if (is.na(total_protein)) {
        showNotification(paste0("Invalid total protein, please check that total protein is a numeric value"), type="error")
        qc_check_ok = F
      }
      if (sum(is.na(y))>0) {
        showNotification(paste0("Invalid values in SPE curve, please check input data"), type="error")
        qc_check_ok = F
      }
      if (length(y)<250) {
        showNotification(paste0("Please enter a SPE curve of at least 250 values"), type="error")
        qc_check_ok = F
      }
      if (!qc_check_ok) {
        printd('server/csf_file:out(inconsistent)')
        showNotification(paste0("Unable to load text input data"), type="error")
        return()
      }
      # create json data file
      temp_id = paste0(session$request$REMOTE_ADDR,'-',round(as.numeric(Sys.time())*1000))
      temp_json_path = paste0(py_temp_path,"/",temp_id,"_temp.json")
      batch_full_id=temp_id
      json <- paste0('{\n\t"batch": {\n\t\t"batch_id": "', batch_full_id, '",\n\t\t"samples": [')
      json <- paste0(json, "\n\t\t{")
      json <- paste0(json, '\n\t\t\t"sample_id": "', sample_id,'",')
      json <- paste0(json, '\n\t\t\t"sex": "', sex,'",')
      json <- paste0(json, '\n\t\t\t"age": ', age, ',')
      json <- paste0(json, '\n\t\t\t"total_protein": ', total_protein, ',')
      json <- paste0(json, '\n\t\t\t"original_curve_y": [', paste(y, collapse = ','), ']')
      json <- paste0(json, "\n\t\t}")
      json <- paste0(json, ']\n\t}\n}')
      write(json, temp_json_path)
      # load json file
      loadJSONBatch(temp_json_path)
    }
    printd('server/single_file_text_ok:out')
  })

  # When we click on download demo template
  output$download_json_template <- downloadHandler(
    filename = function() {
      "spectr_template.json"
    },
    content = function(file) {
      printd('server/download_json_template')
      # write(paste(readLines(demo_json_template_path), collapse="\n"), file)
      file.copy(demo_json_template_path, file)
    }
  )
  
  # When we click on download demo template
  output$download_img_template <- downloadHandler(
    filename = function() {
      "spectr_template.png"
    },
    content = function(file) {
      printd('server/download_img_template')
      # image_write(image_read(demo_img_template_path), file)
      file.copy(demo_img_template_path, file)
    }
  )
  
  # When we click on download demo template
  output$download_img_tutorial <- downloadHandler(
    filename = function() {
      "SPECTR_ImageCroppingTutorial.pdf"
    },
    content = function(file) {
      printd('server/download_img_tutorial')
      file.copy(demo_img_tutorial, file)
    }
  )
  
  # When we click on reload results
  observeEvent(input$reload_results, {
    printd('server/reload_results:in')
    
    now <- as.numeric(Sys.time())*1000
    if (now-last_input>DELAY_BETWEEN_INPUTS) {
      last_input <<- now
    } else {
      printd('server/reload_results:out(toosoon)')
      return()
    }
    
    if (!is.null(buffered_data$batch_data)) {
      printd('server/reload_results:out(alreadyloaded')
      return()
    }
    
    if (USE_SHINYJS) {
      disable('load_demo_batch')
      disable('load_expert_batch')
      disable('load_expert_batch2')
      disable('auto_analyze_input')
      disable('single_upload')
      disable('batch_upload')
      disable('img_file_upload')
      disable('img_file_modal')
      disable('single_file_text_ok')
      disable('download_json_template')
      disable('download_img_template')
      disable('download_img_tutorial')
      disable('reload_results')
      disable('save_button')
    }

    buffered_data$batch_data <- NULL
    buffered_data$current_sample <- 0
    buffered_data$cursor_status <- 0
    buffered_data$spike_start <- 0
    buffered_data$miniatures <- NULL
    
    theFile <- input$reload_results
    if(!is.null(theFile)) {
      # theFile$datapath='C:/Users/hit47/Downloads/batch_1_220319030845.sao'
      tmp <- fromJSON(paste(readLines(theFile$datapath), collapse=""))
      
      # check version
      if (tmp$main_version == MAIN_VERSION & tmp$sub_version == SUB_VERSION) {
        # adapt variables format (list -> data.frame)
        withProgress(message = 'Loading batch...', value = 0, {
          total_n=length(tmp$batch$samples)
          for(i in 1:total_n) {
            incProgress(1/total_n, detail=paste0(i, '/', total_n))
            tmp$batch$samples[[i]]$spikes <- data.frame(index=as.numeric(tmp$batch$samples[[i]]$spikes$index),
                                                        confidence=as.numeric(tmp$batch$samples[[i]]$spikes$confidence),
                                                        start=as.numeric(tmp$batch$samples[[i]]$spikes$start),
                                                        end=as.numeric(tmp$batch$samples[[i]]$spikes$end),
                                                        qty_pct=as.numeric(tmp$batch$samples[[i]]$spikes$qty_pct),
                                                        qty_abs=as.numeric(tmp$batch$samples[[i]]$spikes$qty_abs),
                                                        loc=as.numeric(tmp$batch$samples[[i]]$spikes$loc))
            tmp$batch$samples[[i]]$computer_analysis$predicted_spikes <- data.frame(index=as.numeric(tmp$batch$samples[[i]]$computer_analysis$predicted_spikes$index),
                                                                                    confidence=as.numeric(tmp$batch$samples[[i]]$computer_analysis$predicted_spikes$confidence),
                                                                                    start=as.numeric(tmp$batch$samples[[i]]$computer_analysis$predicted_spikes$start),
                                                                                    end=as.numeric(tmp$batch$samples[[i]]$computer_analysis$predicted_spikes$end),
                                                                                    qty_pct=as.numeric(tmp$batch$samples[[i]]$computer_analysis$predicted_spikes$qty_pct),
                                                                                    qty_abs=as.numeric(tmp$batch$samples[[i]]$computer_analysis$predicted_spikes$qty_abs),
                                                                                    loc=as.numeric(tmp$batch$samples[[i]]$computer_analysis$predicted_spikes$loc))
            
            
            print(paste0('Reloading element ',i))
            if (is.list(tmp$batch$samples[[i]]$original_curve_y)) {
              printd('Original curve y is not list!', level=1)
              # convert to numeric
              tmpel <- tmp$batch$samples[[i]]$original_curve_y
              numeric_values <- numeric(0)
              for(j in 1:length(tmpel)) {
                printd(paste0('Converting value ', j), level=1)
                numeric_values <- c(numeric_values, tmpel[[j]])
              }
              print(paste0('Successfully converted list of size ', length(tmpel), ' to vector of size ', length(numeric_values)))
              tmp$batch$samples[[i]]$original_curve_y <- numeric_values
            }

            # print(tmp$batch$samples[[i]]$boundaries)
            if (is.null(tmp$batch$samples[[i]]$boundaries)) {
              printd('Boundaries were NULL, replacing', level=1)
              tmp$batch$samples[[i]]$boundaries <- numeric(0)
            } else {
              printd('Boundaries were NOT null, ok', level=2)
              # print(tmp$batch$samples[[i]]$boundaries)
            }
            # print(tmp$batch$samples[[i]]$boundaries)
            # compute first analysis
            # print(paste0('Analyzing reloaded sample ',i))
            tmp$batch$samples[[i]]=sampleAnalysis(tmp$batch$samples[[i]], 0, 0, i, usedRecommendations, DEBUG_MODE, FALSE) # without computing plots
          }
          buffered_data$batch_data <- list()
          buffered_data$batch_data$batch = tmp$batch
          buffered_data$current_sample <- 1
          printd('loaded old batch!')
          # printd(buffered_data$batch_data$batch)
          # updateNumericInput(session, "sample_index_input", value = 1,
          #                    min = 1, max = length(tmp$batch$samples), step = 1)
          refreshInputUI()
        })
      } else {
        # show modal
        showModal(modalDialog(
          title = "Unable to load analysis results",
          paste0("File version is not compatible with software version."),
          easyClose = TRUE,
          footer = NULL))
      }     
    }
    if (USE_SHINYJS) {
      enable('load_demo_batch')
      enable('load_expert_batch')
      enable('load_expert_batch2')
      enable('auto_analyze_input')
      enable('single_upload')
      enable('batch_upload')
      enable('img_file_upload')
      enable('img_file_modal')
      enable('single_file_text_ok')
      enable('download_json_template')
      enable('download_img_template')
      enable('download_img_tutorial')
      enable('reload_results')
      enable('save_button')
    }
    
    if (!is.null(buffered_data$batch_data)) {
      updateTabsetPanel(session, "main_tabs", selected = "tab_sample")
    }
    printd('server/reload_results:out')
  })

  observeEvent(input$discard_button, {
    printd('server/discard_button:in')
    
    now <- as.numeric(Sys.time())*1000
    if (now-last_input>DELAY_BETWEEN_INPUTS) {
      last_input <<- now
    } else {
      printd('server/discard_button:out(toosoon)')
      return()
    }
    
    if (is.null(buffered_data$batch_data))
      return()

    # discard batch
    buffered_data$batch_data <- NULL
    buffered_data$current_sample <- 0
    buffered_data$cursor_status <- 0
    buffered_data$spike_start <- 0
    
    printd('server/discard_button:out')
  })

  # prevent inputs modification while loading a new sample
  preventInput <- function() {
    printd('server/preventInput:in')
    if (USE_SHINYJS) {
      disable('prev_button')
      disable('next_button')
      disable('first_button')
      disable('last_button')
      disable('sample_index_input')
      disable('reset_button')
      disable('class_input')
      disable('recom_input')
      disable('haemolysis_input')
      disable('predmap_input')
    }
    printd('server/preventInput:out')
  }
  
  # allow input modification again
  allowInput <- function() {
    printd('server/allowInput:in')
    if (USE_SHINYJS) {
      enable('recom_input')
      enable('prev_button')
      enable('next_button')
      enable('first_button')
      enable('last_button')
      enable('sample_index_input')
      if (buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]$locked==0) { # unlocked
        enable(id = 'class_input')
        enable(id = 'haemolysis_input')
        enable(id = 'valid_button')
        enable(id = 'reset_button')
        enable(id = 'predmap_input')
      } else { # locked
        enable(id = 'invalid_button')
      }
    }

    printd('server/allowInput:in')
  }

  # prev button
  observeEvent(input$prev_button, {
    printd('server/prev_button:in', level=1)
    
    now <- as.numeric(Sys.time())*1000
    printd(paste0('server/prev_button:elapsed -> ', (now-last_input)), level=1)
    if (now-last_input>DELAY_BETWEEN_INPUTS) {
      last_input <<- now
    } else {
      printd('server/prev_button:out(toosoon)', level=1)
      return()
    }

    if (is.null(buffered_data$batch_data))
      return()

    preventInput()
    
    if (buffered_data$current_sample > 1) {
      printd(paste0('server/next_button: switching to prev sample (',buffered_data$current_sample - 1,')'), level=1)
      buffered_data$current_sample <- buffered_data$current_sample - 1
      refreshInputUI()
    }
  
    if (USE_SHINYJS) {
      delay(500, allowInput())
    } else {
      allowInput()
    }
    printd('server/prev_button:out', level=1)
  })
  
  # next button
  observeEvent(input$next_button, {
    printd('server/next_button:in', level=1)
    
    now <- as.numeric(Sys.time())*1000
    printd(paste0('server/next_button:elsaped -> ', (now-last_input)), level=1)
    if (now-last_input>DELAY_BETWEEN_INPUTS) {
      last_input <<- now
    } else {
      printd('server/next_button:out(toosoon)', level=1)
      return()
    }
    
    if (is.null(buffered_data$batch_data))
      return()

    withProgress(message = 'Working...', value = NULL, {
      preventInput()
      
      if (buffered_data$current_sample < length(buffered_data$batch_data$batch$samples)) {
        printd(paste0('server/next_button: switching to next sample (',buffered_data$current_sample + 1,')'), level=1)
        buffered_data$current_sample <- buffered_data$current_sample + 1
        refreshInputUI()
      }
      
      if (USE_SHINYJS) {
        delay(500, allowInput())
      } else {
        allowInput()
      }
    })
    
    printd('server/next_button:out', level=1)
  })
  
  # first button
  observeEvent(input$first_button, {
    printd('server/first_button:in')
    
    now <- as.numeric(Sys.time())*1000
    if (now-last_input>DELAY_BETWEEN_INPUTS) {
      last_input <<- now
    } else {
      printd('server/first_button:out(toosoon)')
      return()
    }
    
    if (is.null(buffered_data$batch_data))
      return()
    
    withProgress(message = 'Working...', value = NULL, {
      preventInput()
      
      buffered_data$current_sample <- 1
      refreshInputUI()
      
      if (USE_SHINYJS) {
        delay(500, allowInput())
      } else {
        allowInput()
      }
    })
    
    printd('server/first_button:out')
  })
  
  # last button
  observeEvent(input$last_button, {
    printd('server/last_button:in')
    
    now <- as.numeric(Sys.time())*1000
    if (now-last_input>DELAY_BETWEEN_INPUTS) {
      last_input <<- now
    } else {
      printd('server/last_button:out(toosoon)')
      return()
    }
    
    if (is.null(buffered_data$batch_data))
      return()
    
    withProgress(message = 'Working...', value = NULL, {
      preventInput()
      
      buffered_data$current_sample <- length(buffered_data$batch_data$batch$samples)
      refreshInputUI()
      
      if (USE_SHINYJS) {
        delay(500, allowInput())
      } else {
        allowInput()
      }
    })
    
    printd('server/last_button:out')
  })
  
  # numeric input
  observeEvent(input$sample_index_input, {
    printd('server/sample_index_input:in', level=1)
    
    now <- as.numeric(Sys.time())*1000
    if (now-last_input>DELAY_BETWEEN_INPUTS) {
      last_input <<- now
    } else {
      printd('server/sample_index_input:out(toosoon)', level=1)
      return()
    }
    
    now <- as.numeric(Sys.time())*1000
    printd(paste0('server/class_input:elapsed(refresh) -> ', (now-last_refresh)), level=1)
    if (now-last_refresh<=DELAY_AFTER_REFRESH) {
      printd('server/sample_index_input:out(toosoon after refresh)', level=1)
      refreshInputUI()
      return()
    }
    
    if (is.null(buffered_data$batch_data)) {
      printd('server/sample_index_input:out(null)', level=1)
      return()
    }
    
    # printd(paste0('updated numeric input: ', buffered_data$current_sample, ' to ', input$sample_index_input))
    if (!is.numeric(input$sample_index_input)) {
      printd('server/sample_index_input:out(notnumeric)', level=1)
      return()
    }
    
    withProgress(message = 'Working...', value = NULL, {
      # store value
      slcted_sample <- input$sample_index_input
      printd(paste0("server/sample_index_input -> sample_index_input is: '",slcted_sample,"' while buffered is: '",buffered_data$current_sample,"'"), level=1)
  
      if (slcted_sample > 0 & slcted_sample <= length(buffered_data$batch_data$batch$samples) & slcted_sample != buffered_data$current_sample) {
        printd(paste0('server/sample_index_input:changing buffered sample to ',slcted_sample), level=1)
        preventInput()
        
        buffered_data$current_sample <- slcted_sample
        refreshInputUI()
        
        if (USE_SHINYJS) {
          delay(500, allowInput())
        } else {
          allowInput()
        }
      }
    })
    
    printd('server/sample_index_input:out', level=1)
  })
  
  # reset button
  observeEvent(input$reset_button, {
    printd('server/reset_button:in')
    
    now <- as.numeric(Sys.time())*1000
    if (now-last_input>DELAY_BETWEEN_INPUTS) {
      last_input <<- now
    } else {
      printd('server/reset_button:out(toosoon)')
      return()
    }
    
    if (is.null(buffered_data$batch_data))
      return()
    
    withProgress(message = 'Working...', value = NULL, {
      # load sample
      tmp <- buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]
      # reset everything
      tmp$spikes <- tmp$computer_analysis$predicted_spikes
      tmp$boundaries <- tmp$computer_analysis$predicted_boundaries
      tmp$class <- tmp$computer_analysis$predicted_class
  
      # store & save
      buffered_data$batch_data$batch$samples[[buffered_data$current_sample]] <- tmp
      
      # recall
      refreshAnalysis()
      refreshInputUI()
    })
    
    printd('server/reset_button:out')
  })
  
  # Output error and warning flags
  output$auto_analysis_output <- renderUI({
    printd('server/auto_analysis_output:in')
    
    now <- as.numeric(Sys.time())*1000
    if (now-last_input>DELAY_BETWEEN_INPUTS) {
      last_input <<- now
    } else {
      printd('server/auto_analysis_output:out(toosoon)')
      return()
    }

    if (is.null(buffered_data$batch_data)) {
      printd('server/auto_analysis_output:out(null)')
      return(NULL)
    }
    
    # Load data
    tmp <- buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]
    
    text_output = character(0)
    
    # fractioning errors
    text_output[length(text_output)+1]=paste0('&#8226; Fractioning confidence: ',format(round(100*tmp$computer_analysis$predicted_fractions_confidence,2),nsmall=2),'%')
    if (tmp$computer_analysis$predicted_fractions_uncertainty==1) {
      text_output[length(text_output)+1]='<span style="color:red">&#8226; Uncertainty about fractioning: monoclonal spike between two fractions?</span>'
    }
    
    # chosen class
    top_class = tmp$computer_analysis$predicted_class
    top_conf = tmp$computer_analysis$predicted_class_confidence

    class_text = paste0('&#8226; Classified as: ', available_classes_input[top_class], ' (', round(100*top_conf, 2), '%)')
    if (top_class != 1 | top_conf < .9)
      class_text <- paste0('<span style="color:red">', class_text, '</span>')
    text_output[length(text_output)+1] <- class_text
    
    s <- tmp$computer_analysis$predicted_spikes
    
    if (nrow(s) > 0) {
      for(i in 1:nrow(s)) {
        text_output[length(text_output)+1] <- paste0('<span style="color:red">&#8226; Monoclonal spike? (', round(s$confidence[i]*100, 2), ' %)</span>')
      }
    } else {
      # No spikes detected by the spike loc model, check judgement of fractioning & classif model
      # Flags based on peak detection for fractioning model & classification model
      if (any(tmp$computer_analysis$predmap_f2 < 0.1)) {
        text_output[length(text_output)+1] <- paste0('<span style="color:red">&#8226; Monoclonal spike? (detected by fractioning model)</span>')
      } else if (top_class!=3 & tmp$computer_analysis$predicted_class_confidences[3]>0.1) {
        text_output[length(text_output)+1] <- paste0('<span style="color:red">&#8226; Monoclonal spike? (detected by classification model) </span>')
      }
    }
    
    # haemolysis
    if (tmp$computer_analysis$predicted_haemolysis != 0)
      text_output[length(text_output)+1] <- paste0('<span style="color:red">&#8226; Hemolysis? (', round(tmp$computer_analysis$predicted_haemolysis_score*100, 2), '%)</span>')
    
    # flags
    if (tmp$computer_analysis$adjustment_flag != 0)
      text_output[length(text_output)+1] <- '<span style="color:red">&#8226; Inconsistent fraction segmentation?</span>'
    if (tmp$computer_analysis$bisalbuminemia_flag != 0)
      text_output[length(text_output)+1] <- '<span style="color:red">&#8226; Bisalbuminemia?</span>'
    
    printd('server/auto_analysis_output:out')
    
    return(HTML(paste(text_output, collapse = '<br>')))
  })
  
  observeEvent(input$class_input, {
    printd('server/class_input:in', level=1)
    
    now <- as.numeric(Sys.time())*1000
    printd(paste0('server/class_input:elapsed -> ', (now-last_input)), level=1)
    if (now-last_input>DELAY_BETWEEN_INPUTS) {
      last_input <<- now
    } else {
      printd('server/class_input:out(toosoon)', level=1)
      return()
    }
    
    now <- as.numeric(Sys.time())*1000
    printd(paste0('server/class_input:elapsed(refresh) -> ', (now-last_refresh)), level=1)
    if (now-last_refresh<=DELAY_AFTER_REFRESH) {
      printd('server/class_input:out(toosoon after refresh)', level=1)
      refreshInputUI()
      return()
    }
    
    # the user changed the class
    # change the class accordingly in tmp
    if (is.null(buffered_data$batch_data)) {
      printd('server/class_input:out(null)', level=1)
      return(FALSE)
    }
    
    withProgress(message = 'Working...', value = NULL, {
      tmp <- buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]
      
      # get the correct index
      chosen_class <- input$class_input
      chosen_class <- gsub(' .+$', '', chosen_class)
      chosen_class_i = which(available_classes_input_short == chosen_class)
      
      if (tmp$class!=chosen_class_i) {
        printd(paste0('server/class_input: class input has CHANGED, stored: ', tmp$class, ', new: ', chosen_class_i), level=1)
      } else {
        printd(paste0('server/class_input: class input has NOT changed, stored: ', tmp$class), level=1)
      }
      
      if (tmp$class != chosen_class_i) {
        printd('server/class_input:class_input changed', level=1)
        if (chosen_class_i != 3) { # we can't just set class to clonal anomaly, we have to add a spike instead
          # change in buffer
          tmp$class <- chosen_class_i
          if (tmp$class != 3) # if class != clonal anomaly, we have to remove all existing spikes
            tmp$spikes <- tmp$spikes[numeric(0),]
          
          buffered_data$batch_data$batch$samples[[buffered_data$current_sample]] <- tmp
          # refresh UI
          refreshAnalysis()
          # printd('calling refreshInputUI from class input')
        }
        refreshInputUI()
      } else {
        printd('server/class_input:class_input DIT NOT change', level=1)
      }
    })

    printd('server/class_input:out', level=1)
  })
  
  observeEvent(input$haemolysis_input, {
    printd('server/haemolysis_input:in', level=2)
    
    now <- as.numeric(Sys.time())*1000
    if (now-last_input>DELAY_BETWEEN_INPUTS) {
      last_input <<- now
    } else {
      printd('server/haemolysis_input:out(toosoon)', level=2)
      return()
    }
    
    now <- as.numeric(Sys.time())*1000
    printd(paste0('server/class_input:elapsed(refresh) -> ', (now-last_refresh)), level=1)
    if (now-last_refresh<=DELAY_AFTER_REFRESH) {
      printd('server/haemolysis_input:out(toosoon after refresh)', level=1)
      refreshInputUI()
      return()
    }
    
    # the user changed the haemolysis status
    # change the class accordingly in tmp
    if (is.null(buffered_data$batch_data)) {
      printd('server/haemolysis_input:out(null)', level=2)
      return(NULL)
    }
    
    withProgress(message = 'Working...', value = NULL, {
      tmp <- buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]
      
      # change in buffer
      tmp$haemolysis = input$haemolysis_input*1
      
      buffered_data$batch_data$batch$samples[[buffered_data$current_sample]] <- tmp
      # refresh UI
      refreshAnalysis()
      refreshInputUI()
    })

    printd('server/haemolysis_input:out', level=2)
  })
  
  # Some backend functions
  refreshAnalysis <- function() {
    printd('server/refreshAnalysis:in')
    
    if (is.null(buffered_data$batch_data)) {
      printd('server/refreshAnalysis:out(null)')
      return(NULL)
    }

    # get old data
    tmp <- buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]
    withProgress(message = 'Working...', value = NULL, {
      # refresh and replace
      buffered_data$batch_data$batch$samples[[buffered_data$current_sample]] <- sampleAnalysis(tmp, buffered_data$cursor_status, buffered_data$spike_start, buffered_data$current_sample, usedRecommendations, DEBUG_MODE, TRUE)
    })
    printd('server/refreshAnalysis:out')
  }
  
  refreshInputUI <- function() {
    printd('server/refreshInputUI:in', level=1)

    tmp <- buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]
  
    if (is.null(tmp)) {
      printd('server/refreshInputUI:out(null)', level=1)
      return()
    }
    
    last_refresh <<- as.numeric(Sys.time())*1000
  
    chosen_class <- input$class_input
    chosen_class <- gsub(' .+$', '', chosen_class)
    chosen_class_i = which(available_classes_input_short == chosen_class)
    
    printd(paste0('server/refreshInputUI: class input selected is: ', chosen_class_i), level=1)

    # change manual input flags for curve
    classes_with_conf = paste0(available_classes_input, ' (', round(100*tmp$computer_analysis$predicted_class_confidences, 2), '%)')
  
    printd(paste0('server/refreshInputUI: class input is set to stored: ', tmp$class), level=1)
    
    last_input <<- as.numeric(Sys.time())*1000
    
    updateSelectInput(session, 'class_input', label = 'Qualitative abnormality:', choices = classes_with_conf,
                      selected = classes_with_conf[tmp$class])
    updateCheckboxInput(session, 'haemolysis_input', value = tmp$haemolysis)
  
    # also refresh number in numeric input
    # update indexer
    updateNumericInput(session, "sample_index_input", value = buffered_data$current_sample,
                       min = 1, max = length(buffered_data$batch_data$batch$samples), step = 1)

    last_refresh <<- as.numeric(Sys.time())*1000

    printd('server/refreshInputUI:out', level=1)
  }
  
  # Output patient's metatata
  output$patient_metadata <- renderText({
    printd('server/patient_metadata:in')
    if (is.null(buffered_data$batch_data)) {
      printd('server/patient_metadata:out(null)')
      return(NULL)
    }
    
    tmp <- buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]
    
    i <- buffered_data$current_sample
    n <- length(buffered_data$batch_data$batch$samples)
    
    printd('server/patient_metadata:out')
    return(HTML(paste0(i, '/', n, ': ', tmp$sample_id, ' (', tmp$sex, ', ', tmp$age, ')<br><br>')))
  })
  
  output$recom_display <- renderText({
    input$recom_input # make call in order to refresh this display if recom_input value is changed
    return(HTML(paste0("Using recommendations from <strong>", usedRecommendations$name, '</strong>. <strong style="color:#FF0000">For research use only.</strong><br>')))
  })

  # Output patient's metatata
  output$sample_id <- renderUI({
    printd('server/sample_id:in')
    if (is.null(buffered_data$batch_data)) {
      printd('server/sample_id:out(null)')
      return(NULL)
    }
    
    tmp <- buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]
    
    printd('server/sample_id:out')
    return(HTML(paste0('<b>Sample ID:</b> ', tmp$sample_id)))
  })

  output$patient_sex <- renderUI({
    printd('server/patient_sex:in')
    if (is.null(buffered_data$batch_data)) {
      printd('server/patient_sex:out(null)')
      return(NULL)
    }
    
    tmp <- buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]
    
    printd('server/patient_sex:out')
    return(HTML(paste0('<b>Sex:</b> ', tmp$sex)))
  })
  
  output$patient_age <- renderUI({
    printd('server/patient_age:in')
    if (is.null(buffered_data$batch_data)) {
      printd('server/patient_age:out(null)')
      return(NULL)
    }
    
    tmp <- buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]
    
    printd('server/patient_age:out')
    return(HTML(paste0('<b>Age:</b> ', tmp$age)))
  })
  
  output$patient_tp <- renderUI({
    printd('server/patient_tp:in')
    if (is.null(buffered_data$batch_data)) {
      printd('server/patient_tp:out(null)')
      return(NULL)
    }
    
    tmp <- buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]
    
    printd('server/patient_tp:out')
    return(HTML(paste0('<b>Total protein:</b> ', tmp$total_protein, ' g/L')))
  })
  
  # Output flags as text
  output$spe_flags <- renderUI({
    printd('server/spe_flags:in')
    
    input$sample_index_input
    input$class_input
    input$haemolysis_input
    
    # Escape if no buffered data
    if (is.null(buffered_data$batch_data)) {
      printd('server/spe_flags:out(null)')
      return(NULL)
    }
    printd('server/spe_flags:out')
    HTML(buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]$cache$flagstext)
  })
  
  # Click on plot
  observeEvent(input$main_plot_click, {
    printd('server/main_plot_click')
    
    # Get x and y coordinates on plot
    x = input$main_plot_click$x
    y = input$main_plot_click$y
    
    if (is.null(x) | is.null(y) | is.null(buffered_data$batch_data))
      return(NULL)
    
    # Load data
    tmp <- buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]
    
    if (tmp$locked == 1)
      return(NULL)
    
    x = round(x)
    if (x < 1 | x > length(tmp$original_curve_y))
      return(NULL)
    
    # Check if currently placing a spike
    if (buffered_data$cursor_status == 1) {
      # finish spike
      # we have to make sure the new spike is at least 3 pixels arge
      # and points are in the right order
      x1 <- buffered_data$spike_start
      x2 <- x
      # sort if x1 > x2
      if (x1 > x2) {
        tmp_x <- x2
        x2 <- x1
        x1 <- tmp_x
      }
      # check length
      x2 <- max(x2, x1+2)
      tmp$spikes[nrow(tmp$spikes)+1, ] <- c(0, 0, x1, x2, 0, 0, 0)
      # recompute spikes order
      tmp$spikes$index <- 1:nrow(tmp$spikes)
      # reset cursor state
      buffered_data$cursor_status <- 0
      buffered_data$spike_start <- 0
      # by the way, the class has to be "clonal anomaly" if there's at least 1 spike
      tmp$class <- 3
      # store & save
      buffered_data$batch_data$batch$samples[[buffered_data$current_sample]] <- tmp
      # recall
      refreshAnalysis()
      refreshInputUI()
      # end
      return()
    }
    
    # Check if click on a boundary
    f <- tmp$boundaries
    clicked_boundary <- which(abs(x-f) < 3)[1]
    if (!is.na(clicked_boundary)) {
      # remove boundary
      f <- f[-clicked_boundary]
      
      # store & save
      tmp$boundaries <- f
      buffered_data$batch_data$batch$samples[[buffered_data$current_sample]] <- tmp
      
      # recompute fractions
      refreshAnalysis()

      # only one action per click
      return()
    }
    
    # if nothing happened : place new boundary
    f <- tmp$boundaries
    f <- c(f, x)
    f <- f[order(f)]
    
    # store & save
    tmp$boundaries <- f
    buffered_data$batch_data$batch$samples[[buffered_data$current_sample]] <- tmp
    
    # recompute fractions
    refreshAnalysis()

    return()
  })
  
  # observeEvent(input$load_demo_batch_hover, {
  #   showNotification("Mouseover on button!")    
  # })
  
  # Double-click on plot
  observeEvent(input$main_plot_dblclick, {
    printd('server/main_plot_dblclick')
    
    # Get x and y coordinates on plot
    x = input$main_plot_dblclick$x
    y = input$main_plot_dblclick$y
    
    if (is.null(x) | is.null(y) | is.null(buffered_data$batch_data))
      return(NULL)
    
    # Load data
    tmp <- buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]
    
    if (tmp$locked == 1)
      return(NULL)
    
    x = round(x)
    if (x < 1 | x > length(tmp$original_curve_y))
      return(NULL)
    
    # Check if currently placing a spike
    if (buffered_data$cursor_status == 1) {
      # cancel spike
      buffered_data$cursor_status = 0
      buffered_data$spike_start <- 0
      refreshAnalysis()
      return()
    }
    
    # Check if currently double clicking on an existing spike
    # cat('\nChecking whether double clicked on spike:\n')
    # cat(paste0('    x: ', x, ', y: ', y, '\n'))
    # cat(paste0('    ', nrow(tmp$spikes), ' spikes, starts: ', tmp$spikes$start, ' ; ends: ', tmp$spikes$end, '\n'))
    # cat(paste0('    y at curve point: ', tmp$original_curve_y[x], '\n'))
    # cat(paste0('    vector of truth: ', paste(tmp$spikes$start <= x & tmp$spikes$end >= x, collapse = ', '), '\n'))
    if (nrow(tmp$spikes) > 0 & y <= tmp$original_curve_y[x]) {
      selected_spike <- which(tmp$spikes$start <= x & tmp$spikes$end >= x)[1]
      # cat(paste0('        Selected spike -> ', selected_spike, '\n'))
      if (!is.na(selected_spike)) {
        # remove spike
        tmp$spikes <- tmp$spikes[-selected_spike, ]
        # recompute order of spikes
        if (nrow(tmp$spikes) > 0) {
          tmp$spikes$index <- 1:nrow(tmp$spikes)
        } else {
          # if no more spikes -> change class
          # by default, we'll chose the top class excluding clonal anomaly, according to CNN results
          corr_pred <- tmp$computer_analysis$predicted_class_confidences
          corr_pred[3] <- 0
          new_top_class = which.max(corr_pred)
          tmp$class <- new_top_class
        }
        # store & save
        buffered_data$batch_data$batch$samples[[buffered_data$current_sample]] <- tmp
        # recall functions
        refreshAnalysis()
        refreshInputUI()
        return()
      }
    }
    
    # Else: start a new spike if enough room
    if (nrow(tmp$spikes) < 4) {
      buffered_data$cursor_status <- 1
      buffered_data$spike_start <- x
      refreshAnalysis()
      return()
    }
    return(NULL)
  })

  # Output final comment if available
  output$final_comment <- renderUI({
    printd('server/final_comment:in')
    
    if (is.null(buffered_data$batch_data)) {
      printd('server/final_comment:out(null)')
      return(NULL)
    }
    
    if (!input$auto_analyze_input) {
      printd('server/final_comment:out(null[expertmode])')
      return(NULL)
    }
    
    comment <- buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]$comment
    # Replace special characters ## no need, apostrophes were "’", not "'"
    # printd(paste0('Comment before replacing apostrophes: "',comment,'"'))
    # comment=gsub("'","&apos;",comment)
    # printd(paste0('Comment after replacing apostrophes: "',comment,'"'))
    comment <- paste0('&#8226; ', comment)
    comment <- gsub('\n', '<br>&#8226; ', comment)
    printd('server/final_comment:out')
    HTML(comment)
  })
  
  output$quantification <- renderDT({
    printd('server/final_comment:in')
    if (is.null(buffered_data$batch_data)) {
      printd('server/final_comment:out(null)')
      return(NULL)
    }
    printd('server/final_comment:out')
    buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]$cache$fractions_dt
  })
  
  output$spike_quantification <- renderDT({
    printd('server/spike_quantification:in')
    if (is.null(buffered_data$batch_data)) {
      printd('server/spike_quantification:out(null)')
      return(NULL)
    }
    printd('server/spike_quantification:out')
    buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]$cache$spikes_dt
  })
  
  output$main_plot <- renderPlot({
    printd('server/main_plot:in')
    if (is.null(buffered_data$batch_data)) {
      printd('server/main_plot:out(null)')
      return(NULL)
    }
    
    # force computing plots
    tmp <- buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]
    # print('About to draw main plot, force analysis')
    tmp <- sampleAnalysis(tmp, buffered_data$cursor_status, buffered_data$spike_start, buffered_data$current_sample, usedRecommendations, DEBUG_MODE, TRUE)
    # print('Analysis ok, lets plot')
    buffered_data$batch_data$batch$samples[[buffered_data$current_sample]] <- tmp
    # printd(buffered_data$batch_data$batch$samples[[buffered_data$current_sample]])
    
    printd('server/main_plot:out')
    if (input$predmap_input=='Fractions')
      return(buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]$cache$main_plot_with_fractions_predmap)
    else if (input$predmap_input=='Spikes')
      return(buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]$cache$main_plot_with_spikes_predmap)
    else
      return(buffered_data$batch_data$batch$samples[[buffered_data$current_sample]]$cache$main_plot)
    
  })

  output$miniatures_grid <- renderImage({
    printd('server/miniatures_grid:in')
    
    # buffered_data
    # printd('refreshing miniatures grid')
    
    # make a grid of all possible plots
    if (is.null(buffered_data$batch_data)) {
      printd('server/miniatures_grid:out(null)')
      tempfile = image_blank(1,1,"white") %>%
        image_write(tempfile(fileext='jpg'), format = 'jpg')
      return(list(src = tempfile, width='100%'))
    }
    
    # create empty thumbnail
    if (QUICK_MINIATURES) {
      createFakePlot <- function(template_gp, sample_id, batch_index) {
        img <- image_annotate(thumbnail_template, paste0(sample_id,' (',batch_index,')'), size = 15, gravity = "center", color = "#000000", loc="+0+83")
        return(img)
      }
    } else {
      template_gp <- ggplot() +
        geom_text(data = data.frame(x = 150, y = .5, label = "Not loaded yet"), aes(x = x, y = y, label = label), col = '#000000', size = 4) +
        xlim(c(1,304)) +
        ylim(c(0,1)) +
        theme_minimal() +
        theme(axis.line=element_blank(),
              axis.text.x=element_blank(),
              axis.text.y=element_blank(),
              axis.ticks=element_blank(),
              # axis.title.x=element_blank(),
              axis.title.y=element_blank(),
              legend.position="none",
              panel.background=element_blank(),
              panel.border = element_rect(colour = "#333333", fill=NA, size=1),
              # panel.border=element_blank(),
              panel.grid.major=element_blank(),
              panel.grid.minor=element_blank(),
              plot.background=element_blank()) +
        coord_fixed(ratio=SPE_WIDTH)
      
      createFakePlot <- function(template_gp, sample_id, batch_index) {
        gp <- template_gp +
          xlab(paste0(sample_id, ' (', batch_index, ')'))
        # save plot as image
        fig <- image_graph(width = 200, height=200, res=96)
        print(gp)
        dev.off()
        return(fig)
      }
    }
    
    withProgress(message = 'Working...', value = 0, {
      n=length(buffered_data$batch_data$batch$samples)
      
      incProgress(0, detail = "Rendering curves")
      printd('generating grid', level=2)
      printd(paste0('no of images: ',n), level=2)
      n_row=8
      figs=list()
      for (i in 1:n) {
        # printd(image_info(buffered_data$batch_data$batch$samples[[i]]$cache$miniature))
        if (!is.null(buffered_data$batch_data$batch$samples[[i]]$cache$miniature)) {
          printd(paste0('image ',i,' -> adding real miniature'), level=2)
          figs[[i]]=buffered_data$batch_data$batch$samples[[i]]$cache$miniature
        } else {
          printd(paste0('image ',i,' -> adding fake miniature'), level=2)
          figs[[i]]=createFakePlot(template_gp, buffered_data$batch_data$batch$samples[[i]]$sample_id, i)
        }
        incProgress(1/n, detail = "Rendering curves")
      }
      # if not enough figures : add to have at least 8
      if (length(figs)<n_row) {
        printd('adding blank figures to fill', level=2)
        for (i in (length(figs)+1):n_row) {
          figs[[i]]=image_blank(width=200,height=200,"white")
        }
      }
      # img2 <- image_write(fig, path=NULL, format="png")
      rows=list()
      if (n>n_row) {
        for(i in 1:floor(n/n_row)) {
          rows[[i]]=image_append(image_join(figs[((i-1)*n_row+1):(i*n_row)]))
        }
        if (i*n_row<n) {
          rows[[i+1]]=image_append(image_join(figs[(i*n_row+1):(n)]))
        }
        printd('appending rows', level=2)
        res=image_append(image_join(rows), stack=T)
      } else {
        printd('appending figures', level=2)
        res=image_append(image_join(figs))
      }
    })
    printd('saving miniatures', level=2)
    tempfile = res %>%
      image_write(tempfile(fileext='jpg'), format = 'jpg')
    printd('done generating grid', level=2)
    printd('server/miniatures_grid:out')
    return(list(src = tempfile, width='100%'))
    # height=ceiling(length(buffered_data$batch_data$batch$samples)/8)*200, contentType = "image/bmp"
  }, deleteFile=TRUE)
  
  # save batch button
  output$save_button <- downloadHandler(
    filename=function() {
      paste('results_', buffered_data$batch_data$batch$batch_id, '_', format(Sys.time(), format='%d%m%y%H%M%S'), '.json', sep='')
    },
    content=function(file) {
      if (is.null(buffered_data$batch_data))
        return()
      
      # should save now
      buffer_out = list()
      # add metadata
      buffer_out$main_version=MAIN_VERSION
      buffer_out$sub_version=SUB_VERSION
      buffer_out$minor_version=MINOR_VERSION
      buffer_out$time=format(Sys.time())
      # copy batch
      buffer_out$batch=list()
      buffer_out$batch$batch_id=buffered_data$batch_data$batch$batch_id
      buffer_out$batch$samples=list()
      for(i in 1:length(buffered_data$batch_data$batch$samples)) {
        tmp_sample=buffered_data$batch_data$batch$samples[[i]]
        tmp_buffer=list()
        tmp_buffer$sample_id=tmp_sample$sample_id
        # tmp_buffer$sample_batchid=tmp_sample$sample_batchid
        # tmp_buffer$patient_birthdate=tmp_sample$patient_birthdate
        # tmp_buffer$patient_samplingdate=tmp_sample$patient_samplingdate
        # tmp_buffer$fullname=tmp_sample$fullname
        tmp_buffer$sex=tmp_sample$sex
        tmp_buffer$age=tmp_sample$age
        tmp_buffer$total_protein=tmp_sample$total_protein
        # tmp_buffer$prescriber=tmp_sample$prescriber
        tmp_buffer$original_curve_y=tmp_sample$original_curve_y
        tmp_buffer$computer_analysis=tmp_sample$computer_analysis
        tmp_buffer$class=tmp_sample$class
        tmp_buffer$boundaries=tmp_sample$boundaries
        tmp_buffer$spikes=tmp_sample$spikes
        tmp_buffer$custom_comment=tmp_sample$custom_comment
        tmp_buffer$haemolysis=tmp_sample$haemolysis
        tmp_buffer$locked=tmp_sample$locked
        tmp_buffer$bin_analysis=tmp_sample$bin_analysis
        tmp_buffer$fractions_qty_pct=tmp_sample$fractions_qty_pct
        tmp_buffer$fractions_qty_abs=tmp_sample$fractions_qty_abs
        tmp_buffer$fractions_names=tmp_sample$fractions_names
        tmp_buffer$fractions_shortnames=tmp_sample$fractions_shortnames
        tmp_buffer$fractions_longnames=tmp_sample$fractions_longnames
        tmp_buffer$fractions_residual_qty_pct=tmp_sample$fractions_residual_qty_pct
        tmp_buffer$fractions_residual_qty_abs=tmp_sample$fractions_residual_qty_abs
        tmp_buffer$flags=tmp_sample$flags
        tmp_buffer$comment=tmp_sample$comment
        # store new buffer withtout cache and only selected values
        buffer_out$batch$samples[[i]]=tmp_buffer
      }
      # buffer_out$batch = buffered_data$batch_data$batch
      # convert to json
      temp_file = toJSON(buffer_out)
      # export
      write(temp_file, file)
    },
    contentType='text/plain'
  )
  
  # Refreshers
    
  outputOptions(output, "disclaimer_validated", suspendWhenHidden = FALSE)
  outputOptions(output, "currently_active_batch", suspendWhenHidden = FALSE)
  outputOptions(output, "batch_sample_active", suspendWhenHidden = FALSE)
  outputOptions(output, "debug_mode", suspendWhenHidden = FALSE)
  
  options(encoding = "UTF-8")
}

##################################################################################################
############################################ UI ##################################################
##################################################################################################

replacement_text <- paste0("'container': 'body', 'delay': {'show':", POPUPS_DELAY_IN, ", 'hide':", POPUPS_DELAY_OUT, "},")
delayBSPopover <- function(elem) {
  elem$children[[1]] <- gsub("'container': 'body',", replacement_text, elem$children[[1]])
  return(elem)
}

if (USE_SHINYJS) {
  ui=fluidPage(
    useShinyjs(),
    HTML('<br>'),
    # Sidebar with a slider input for number of bins 
    sidebarLayout(
      sidebarPanel(
        # Application title
        titlePanel('SPECTR'),
        conditionalPanel(condition = 'output.batch_sample_active',
                         htmlOutput('recom_display'),
                         tags$div(HTML('<a href="mailto:floris.chabrun@chu-angers.fr">E-mail the corresponding author</a><br><br>')), id="email_div"),
        conditionalPanel(condition = '!output.batch_sample_active',
                         HTML(paste0('Serum Protein Electrophoresis computer-assisted recognition v', MAIN_VERSION, '.', SUB_VERSION, '.', MINOR_VERSION, '<br><br>')),
                         HTML('<strong>F. Chabrun, X. Dieu, D. Prunier-Mirebeau, P. Reynier, Angers University Hospital.</strong><br><br>'),
                         HTML('<strong style="color:#FF0000">For research use only.</strong><br><br>')),
        conditionalPanel(condition = '!output.batch_sample_active',
                         HTML('<strong>Software filed under number IDDN.FR.001.440032.000.S.P.2019.000.31230.</strong><br><br>')),
        conditionalPanel(condition = 'output.disclaimer_validated',
          conditionalPanel(condition = '!output.currently_active_batch',
                           tags$div(selectInput('recom_input', 'Recommendations:', sapply(1:length(RECOMMENDATIONS), function(i) {RECOMMENDATIONS[[i]]$name}), selectize = TRUE), id='recom_div'),
                           conditionalPanel(condition = '!output.currently_active_batch',
                                            HTML('<br>'),
                                            tags$div(strong(HTML("Load the demonstration batch or upload a sample to start analysis."))), id="demo_batch_div"),
                           HTML('<br>'),
                           fluidRow(column(width=12, offset=0, align='center', actionButton("load_demo_batch", "Load demo batch"))),
                           HTML('<br>'),
                           conditionalPanel(condition = 'output.debug_mode',
                                            strong(HTML('Expert test:')),
                                            HTML('<br>'),
                                            actionButton("load_expert_batch", "Load expert batch"),
                                            # actionButton("load_expert_batch2", "Load expert batch B"),
                                            HTML('<br>'),
                                            checkboxInput('auto_analyze_input', 'Automatically analyze samples', !DEBUG_MODE),
                                            fileInput("reload_results", "Reload results")),

                           tags$div(HTML("<strong>Upload single sample</strong><br>"),
                                    fluidRow(column(width=12, offset=0, align='center', actionButton("single_upload", "Upload single sample"))),
                                    HTML("<br>"),
                                    id="single_upload_div"),
                           bsModal("img_file_modal", HTML("<strong>Upload single sample</strong>"), "single_upload", size = "large",
                                   HTML('Fill in the metadata fields below <u>before</u> either loading an image file or filling in the raw curve data as text.<br><br>'),
                                   numericInput("img_file_age", "Age (years):", 50, min=0, max=150, step=.5),
                                   selectInput("img_file_sex", "Sex:", c("F","M"), selected="M"),
                                   numericInput("img_file_tp", "Total protein (g/L):", 70, min=5, max=150, step=1),
                                   HTML("<hr>"),
                                   HTML('<strong>Upload data as:</strong>'),
                                   fluidRow(column(6, HTML("<br>"), fileInput('img_file_upload', 'Image file:', accept = c('.png','.tif','.bmp')),
                                                   HTML("Please download the template below for an example of an image that can be uploaded. <strong style='color:red;'>Please note that the curve extracted from the image may be slightly different from the original curve, which may alter the interpretation.</strong><br><br>"),
                                                   fluidRow(column(6, downloadButton("download_img_template", "Download Image template")),
                                                            column(6, downloadButton("download_img_tutorial", "Download cropping tutorial")), align='center')),
                                            column(6, textAreaInput("single_file_text", label="Direct input:", height="180px", placeholder="0,2,2,1,1,1,1,1,0,1,1,1,2,2,2,2,2,2,2,4,4,4,5,6,7,7,8,10,12,12,14,17,20,24,27,30,31,30,29,26,21,18,13,11,8,7,7,6,6,7,8,10,11,12,12,12,13,14,17,21,27,36,46,59,75,92,112,136,163,194,227,265,308,356,411,478,561,663,794,961,1177,1455,1806,2239,2769,3334,3819,4095,4082,3769,3208,2486,1748,1152,738,476,312,207,137,93,70,57,49,42,36,31,27,25,24,24,25,27,30,33,36,39,44,48,54,61,70,82,96,114,136,155,168,170,162,145,121,96,75,64,67,75,84,94,103,112,118,125,131,137,144,149,151,152,150,147,144,143,145,150,158,166,174,177,178,176,171,165,162,156,152,146,138,127,118,107,99,90,86,81,78,76,74,74,73,73,75,77,81,87,93,101,111,122,137,159,196,247,309,366,401,401,371,315,246,187,151,138,145,155,164,171,180,193,213,244,275,295,300,290,265,232,201,182,172,170,176,184,193,202,214,225,235,246,256,265,275,285,296,306,315,325,333,340,346,352,357,361,363,364,364,360,354,342,327,309,287,263,238,214,191,169,147,128,109,93,77,63,52,43,34,27,23,18,14,11,8,6,5,4,4,2,2,2,2,2,2,2,2,1,1,1,1,1,1,0,0,0,0,0"),
                                                   fluidRow(column(12,actionButton("single_file_text_ok", "Upload sample")), align="right"),
                                                   style='border-left:1px solid silver;'))),
                           tags$div(fileInput('batch_upload', 'Upload sample batch as JSON', accept = c('.json')),
                                    fluidRow(column(width=12, offset=0, align='left', downloadButton("download_json_template", "Download JSON template"))),
                                    id="batch_upload_div")),

          conditionalPanel(condition = '!output.batch_sample_active',
                           conditionalPanel(condition = 'output.currently_active_batch',
                                            conditionalPanel(condition = 'output.debug_mode',
                                                             strong(HTML('Expert test:')),
                                                             HTML('<br>'),
                                                             downloadButton("save_button", "Save results"),
                                                             HTML('<br><br>')),
                                            fluidRow(
                                              column(width = 12,
                                                     offset = 0,
                                                     align='center',
                                                     actionButton("discard_button", "Discard batch"))))),
          conditionalPanel(condition = 'output.batch_sample_active',
                           tags$div(selectInput('predmap_input', 'CNN map visualization:', predmap_classes, selectize = TRUE), id='predmap_input_div'),
                           strong(htmlOutput('patient_metadata')),
                           conditionalPanel(condition = 'output.currently_active_batch',
                                            fluidRow(
                                              column(width = 12,
                                                     offset = 0,
                                                     align='center',
                                                     actionButton("first_button", "<<"),
                                                     actionButton("prev_button", "<"),
                                                     div(style="display: inline-block;vertical-align:top; width: 70px;",numericInput('sample_index_input', NULL, NULL)),
                                                     actionButton("next_button", ">"),
                                                     actionButton("last_button", ">>"),
                                                     actionButton("reset_button", "Reset"))),
                                            tags$div(selectInput('class_input', 'Qualitative abnormality:', available_classes_input, selectize = TRUE), id='class_input_div'),
                                            tags$div(checkboxInput('haemolysis_input', 'Hemolyzed serum', F), id='haemolysis_div'),
                                            tags$div(
                                              strong(HTML('Automated analysis output:')),
                                              htmlOutput('auto_analysis_output'),
                                              id = 'auto_analysis_div'),
                                            HTML('<br>'),
                                            tags$div(
                                              strong(HTML('Analysis flags:')),
                                              htmlOutput('spe_flags'),
                                              id = 'spe_flags_div')))), width = 3),
      mainPanel(
        conditionalPanel(condition = '!output.disclaimer_validated',
                         titlePanel('SPECTR'),
                         HTML(paste0('<strong>Serum Protein Electrophoresis computer-assisted recognition software v', MAIN_VERSION, '.', SUB_VERSION, '.', MINOR_VERSION, '</strong><br>F. Chabrun, X. Dieu, D. Prunier-Mirebeau, P. Reynier, Angers University Hospital.<br><br>')),
                         HTML(LICENSE_TEXT),
                         fluidRow(column(width=9, offset=0, align='center', actionButton("disclaimer_valid", "I have read the terms and fully agree")))),
        conditionalPanel(condition = 'output.currently_active_batch',
                         tabsetPanel(id='main_tabs', type='tabs',
                                     tabPanel('Batch', value='tab_batch', plotOutput('miniatures_grid')),
                                     tabPanel('Sample', value='tab_sample',
                                              column(width = 5,
                                                     HTML('<br>'),
                                                     htmlOutput('sample_id'),
                                                     htmlOutput('patient_age'),
                                                     htmlOutput('patient_sex'),
                                                     htmlOutput('patient_tp'),
                                                     HTML('<br>'),
                                                     dataTableOutput('quantification'),
                                                     HTML('<br>'),
                                                     dataTableOutput('spike_quantification')),
                                              column(width = 7,
                                                     fluidRow(plotOutput('main_plot',
                                                                         width = "100%",
                                                                         click = 'main_plot_click',
                                                                         dblclick = dblclickOpts(id = "main_plot_dblclick"),
                                                                         hover = hoverOpts(id = "main_plot_hover"),
                                                                         brush = brushOpts(id = "main_plot_brush"))),
                                                     tags$div(
                                                       strong(htmlOutput('comment_title')),
                                                       HTML('<br>'),
                                                       htmlOutput('final_comment'),
                                                       id='comment_div'))))), width = 9)),
    delayBSPopover(bsPopover(id = 'email_div', title = "Send an e-mail to the corresponding author", content='Please do not hesitate to contact the corresponding author if you encounter any issue or have feedback for improving of SPECTR.', placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    
    bsPopover(id = 'demo_batch_div', title = "Load demonstration samples", content="Load a demonstration batch of eight samples and run the analysis", placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS),
    delayBSPopover(bsPopover(id = 'single_upload_div', title = "Upload single sample", content='Upload your own sample and run analysis using SPECTR. You may upload your data either as raw numeric values or as an image of the SPE curve.', placement = "top", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    delayBSPopover(bsPopover(id = 'batch_upload_div', title = 'Upload a batch of samples', content = "Upload a batch of your own samples, as a JSON file, and run analysis using SPECTR. Please download the JSON template below for formatting your data.", placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    delayBSPopover(bsPopover(id = 'download_img_template', title = "Download image template", content='Download an image template for uploading an SPE curve as image. Please note the SPE may already be fractioned as pictured in the template image, <strong>but no M-spikes should be placed on the curve.</strong>', placement = "top", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    
    delayBSPopover(bsPopover(id = 'discard_button', title = "Discard batch", content="Unload currently analyzed batch and load another sample/demonstration batch", placement = "bottom", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    delayBSPopover(bsPopover(id = 'predmap_input_div', title = "CNN map visualization", content="Select a model (<b>fractioning</b> model or <b>peak detection</b> model) to <b>overlay the output predictions on the curve</b>", placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    
    delayBSPopover(bsPopover(id = 'first_button', title = "First sample", content="Return to first sample of batch", placement = "top", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    delayBSPopover(bsPopover(id = 'next_button', title = "Next sample", content="Go to next sample of batch", placement = "top", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    delayBSPopover(bsPopover(id = 'last_button', title = "Last sample", content="Go to last sample of batch", placement = "top", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    delayBSPopover(bsPopover(id = 'prev_button', title = "Previous sample", content="Return to previous sample of batch", placement = "top", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    delayBSPopover(bsPopover(id = 'sample_index_input', title = "Go to", content="Jump to another sample", placement = "top", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    
    delayBSPopover(bsPopover(id = 'reset_button', title = "Reset sample analysis", content="Reset all changes applied to the current sample and revert to automatic SPECTR analysis", placement = "bottom", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    
    delayBSPopover(bsPopover(id = 'class_input_div', title = 'Qualitative abnormality class', content = 'Use this list to see the <b>probability output</b> for each <b>qualitative abnormality</b>.<br><br>You may manually force SPECTR to change the classification of the sample by changing the selected class. For classifying the curve as with peak of monoclonal abnormality, directly place a new peak on the curve instead of selecting the "Clonal anomaly" class.<br><br><b>Interpretation will be updated automatically</b> in real time', placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    
    delayBSPopover(bsPopover(id = 'haemolysis_div', title = "Hemolyzed serum", content="Check this if the serum sample shows hemolysis and it should be taken into account in the final interpretation", placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    
    delayBSPopover(bsPopover(id = 'auto_analysis_div', title = "Deep learning analysis output", content="This section lists <b>all abnormalities</b> automatically detected by <b>deep learning models</b>", placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    delayBSPopover(bsPopover(id = 'spe_flags_div', title = "Automatic analysis flags", content="This section lists <b>all flags detected by the expert system</b> (included based on the deep learning analysis output) and <b>used for computing the automatic interpretation</b>", placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    
    delayBSPopover(bsPopover(id = 'miniatures_grid', title = "Batch panel", content="The batch panel lists all samples currently loaded. Samples outlined by a <b>red frame</b> show <b>qualitative or quantitative abnormalities</b>", placement = "left", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    # delayBSPopover(bsPopover(id = 'sample_id', title = "Sample ID", content="Sample identification number", placement = "bottom", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS),
    # delayBSPopover(bsPopover(id = 'patient_age', title = "Age", content="Patient's age at sampling (years)", placement = "bottom", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS),
    # delayBSPopover(bsPopover(id = 'patient_sex', title = "Sex", content="Patient's sex", placement = "bottom", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS),
    delayBSPopover(bsPopover(id = 'patient_tp', title = "Total protein", content="Serum total protein for this sample, used for computing absolute quantities in fractions", placement = "bottom", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    
    delayBSPopover(bsPopover(id = 'quantification', title = "Quantification results", content='<b>Relative</b> ("Pct.") and <b>absolute</b> ("Conc.") <b>quantities</b> in each fraction, with reference values for adults ("Ref. (%)", "Ref. (g/L)", Bato et al. & SEBIA)<br><br>If a peak is placed in the gamma zone, the <b>residual gammaglobulins</b> are computed and showed in this table ("Gamma (res.)")<br><br>The <b>results are automatically updated</b> if the fractioning is changed on the curve', placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    delayBSPopover(bsPopover(id = 'spike_quantification', title = "Peaks quantification", content='<b>Relative</b> ("Pct.") and <b>absolute</b> ("Conc.") <b>quantities</b> in each peak placed on the curve', placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    
    delayBSPopover(bsPopover(id = 'recom_div', title = "Recommendations", content='Select publication used for harmonizing text comments.', placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    
    delayBSPopover(bsPopover(id = 'main_plot', title = "Interactive SPE curve", content='<b>Click</b> to place a new fraction or <b>click</b> on an existing fraction to remove it. Quantification is automatically updated and results will be updated in real time in the quantification table<br><br><b>Double-click</b> on a point, then <b>click</b> on another point to place the start and end points of the new peak (in any order) or <b>double-click</b> on an existing peak to remove it<br><br>Quantification and final interpretation will be <b>automatically updated and displayed in real time</b>', placement = "left", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
    delayBSPopover(bsPopover(id = 'comment_div', title = "Automatic interpretation", content="This <b>comment</b> is based on deep learning analysis and is <b>updated in real time</b> according to user modifications", placement = "left", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)))
# } else {
#   # TODO update
#   ui=fluidPage(
#     # useShinyjs(),
#     HTML('<br>'),
#     # Sidebar with a slider input for number of bins 
#     sidebarLayout(
#       sidebarPanel(
#         # Application title
#         titlePanel('SPECTR'),
#         conditionalPanel(condition = 'output.batch_sample_active',
#                          htmlOutput('recom_display')),
#         conditionalPanel(condition = '!output.batch_sample_active',
#                          HTML(paste0('Serum Protein Electrophoresis computer-assisted recognition v', MAIN_VERSION, '.', SUB_VERSION, '.', MINOR_VERSION, '<br><br>')),
#                          HTML('<strong>F. Chabrun, X. Dieu, D. Prunier-Mirebeau, P. Reynier, Angers University Hospital.</strong><br><br>'),
#                          HTML('<strong style="color:#FF0000">For research use only.</strong><br><br>')),
#         conditionalPanel(condition = '!output.batch_sample_active',
#                          HTML('<strong>Software deposited under number IDDN.FR.001.440032.000.S.P.2019.000.31230.</strong><br><br>')),
#         conditionalPanel(condition = 'output.disclaimer_validated',
#                          conditionalPanel(condition = '!output.currently_active_batch',
#                                           tags$div(selectInput('recom_input', 'Recommendations:', sapply(1:length(RECOMMENDATIONS), function(i) {RECOMMENDATIONS[[i]]$name}), selectize = TRUE), id='recom_div'),
#                                           HTML('<br>'),
#                                           fluidRow(column(width=12, offset=0, align='center', actionButton("load_demo_batch", "Load demo batch"))),
#                                           HTML('<br>'),
#                                           conditionalPanel(condition = 'output.debug_mode',
#                                                            strong(HTML('Expert test:')),
#                                                            HTML('<br>'),
#                                                            actionButton("load_expert_batch", "Load expert batch"),
#                                                            # actionButton("load_expert_batch2", "Load expert batch B"),
#                                                            HTML('<br>'),
#                                                            checkboxInput('auto_analyze_input', 'Automatically analyze samples', !DEBUG_MODE),
#                                                            fileInput("reload_results", "Reload results")),
#                                           tags$div(fileInput('csv_file', 'Upload sample as CSV', accept = c('.csv')), id="csv_file_div"),
#                                           fluidRow(column(width=12, offset=0, align='left', downloadButton("download_csv_template", "Download CSV template")))),
#                          conditionalPanel(condition = '!output.batch_sample_active',
#                                           conditionalPanel(condition = 'output.currently_active_batch',
#                                                            conditionalPanel(condition = 'output.debug_mode',
#                                                                             strong(HTML('Expert test:')),
#                                                                             HTML('<br>'),
#                                                                             downloadButton("save_button", "Save results"),
#                                                                             HTML('<br><br>')),
#                                                            fluidRow(
#                                                              column(width = 12,
#                                                                     offset = 0,
#                                                                     align='center',
#                                                                     actionButton("discard_button", "Discard batch"))))),
#                          conditionalPanel(condition = 'output.batch_sample_active',
#                                           tags$div(selectInput('predmap_input', 'CNN map visualization:', predmap_classes, selectize = TRUE), id='predmap_input_div'),
#                                           strong(htmlOutput('patient_metadata')),
#                                           conditionalPanel(condition = 'output.currently_active_batch',
#                                                            fluidRow(
#                                                              column(width = 12,
#                                                                     offset = 0,
#                                                                     align='center',
#                                                                     actionButton("first_button", "<<"),
#                                                                     actionButton("prev_button", "<"),
#                                                                     div(style="display: inline-block;vertical-align:top; width: 70px;",numericInput('sample_index_input', NULL, NULL)),
#                                                                     actionButton("next_button", ">"),
#                                                                     actionButton("last_button", ">>"),
#                                                                     actionButton("reset_button", "Reset"))),
#                                                            tags$div(selectInput('class_input', 'Qualitative abnormality:', available_classes_input, selectize = TRUE), id='class_input_div'),
#                                                            tags$div(checkboxInput('haemolysis_input', 'Hemolyzed serum', F), id='haemolysis_div'),
#                                                            tags$div(
#                                                              strong(HTML('Automated analysis output:')),
#                                                              htmlOutput('auto_analysis_output'),
#                                                              id = 'auto_analysis_div'),
#                                                            HTML('<br>'),
#                                                            tags$div(
#                                                              strong(HTML('Analysis flags:')),
#                                                              htmlOutput('spe_flags'),
#                                                              id = 'spe_flags_div'))),
#                          conditionalPanel(condition = '!output.currently_active_batch',
#                                           HTML('<br>'),
#                                           strong(HTML("Load the demonstration batch or upload a sample to start analysis.")))), width = 3),
#       mainPanel(
#         conditionalPanel(condition = '!output.disclaimer_validated',
#                          titlePanel('SPECTR'),
#                          HTML(paste0('<strong>Serum Protein Electrophoresis computer-assisted recognition software v', MAIN_VERSION, '.', SUB_VERSION, '.', MINOR_VERSION, '</strong><br>F. Chabrun, X. Dieu, D. Prunier-Mirebeau, P. Reynier, Angers University Hospital.<br><br>')),
#                          HTML(LICENSE_TEXT),
#                          fluidRow(column(width=9, offset=0, align='center', actionButton("disclaimer_valid", "I've read the terms and fully agree")))),
#         conditionalPanel(condition = 'output.currently_active_batch',
#                          tabsetPanel(id='main_tabs', type='tabs',
#                                      tabPanel('Batch', value='tab_batch', plotOutput('miniatures_grid')),
#                                      tabPanel('Sample', value='tab_sample',
#                                               column(width = 5,
#                                                      HTML('<br>'),
#                                                      htmlOutput('sample_id'),
#                                                      htmlOutput('patient_age'),
#                                                      htmlOutput('patient_sex'),
#                                                      htmlOutput('patient_tp'),
#                                                      HTML('<br>'),
#                                                      dataTableOutput('quantification'),
#                                                      HTML('<br>'),
#                                                      dataTableOutput('spike_quantification')),
#                                               column(width = 7,
#                                                      fluidRow(plotOutput('main_plot',
#                                                                          width = "100%",
#                                                                          click = 'main_plot_click',
#                                                                          dblclick = dblclickOpts(id = "main_plot_dblclick"),
#                                                                          hover = hoverOpts(id = "main_plot_hover"),
#                                                                          brush = brushOpts(id = "main_plot_brush"))),
#                                                      tags$div(
#                                                        strong(htmlOutput('comment_title')),
#                                                        HTML('<br>'),
#                                                        htmlOutput('final_comment'),
#                                                        id='comment_div'))))), width = 9)),
#     bsPopover(id = 'load_demo_batch', title = "Load demonstration samples", content="Load a demonstration batch of eight samples and run the analysis", placement = "bottom", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS),
#     # bsPopover(id = 'csv_file_div', title = "Upload sample", content='Upload your own samples and run analysis using SPECTR. Use the "Download CSV template" button to see an example of how your file should be formatted', placement = "top", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS),
#     # delayBSPopover(bsPopover(id = 'download_csv_template', title = "Download CSV template", content="Download a CSV template for formatting your sample in order to upload it and analyze it through SPECTR", placement = "top", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     
#     delayBSPopover(bsPopover(id = 'discard_button', title = "Discard batch", content="Unload currently analyzed batch and load another sample/demonstration batch", placement = "bottom", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     delayBSPopover(bsPopover(id = 'predmap_input_div', title = "CNN map visualization", content="Select a model (<b>fractioning</b> model or <b>peak detection</b> model) to <b>overlay the output predictions on the curve</b>", placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     
#     delayBSPopover(bsPopover(id = 'first_button', title = "First sample", content="Return to first sample of batch", placement = "top", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     delayBSPopover(bsPopover(id = 'next_button', title = "Next sample", content="Go to next sample of batch", placement = "top", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     delayBSPopover(bsPopover(id = 'last_button', title = "Last sample", content="Go to last sample of batch", placement = "top", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     delayBSPopover(bsPopover(id = 'prev_button', title = "Previous sample", content="Return to previous sample of batch", placement = "top", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     delayBSPopover(bsPopover(id = 'sample_index_input', title = "Go to", content="Jump to another sample", placement = "top", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     
#     delayBSPopover(bsPopover(id = 'reset_button', title = "Reset sample analysis", content="Reset all changes applied to the current sample and revert to automatic SPECTR analysis", placement = "bottom", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     
#     delayBSPopover(bsPopover(id = 'class_input_div', title = 'Qualitative abnormality class', content = 'Use this list to see the <b>probability output</b> for each <b>qualitative abnormality</b>.<br><br>You may manually force SPECTR to change the classification of the sample by changing the selected class. For classifying the curve as with peak of monoclonal abnormality, directly place a new peak on the curve instead of selecting the "Clonal anomaly" class.<br><br><b>Interpretation will be updated automatically</b> in real time', placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     
#     delayBSPopover(bsPopover(id = 'haemolysis_div', title = "Hemolyzed serum", content="Check this if the serum sample shows hemolysis and it should be taken into account in the final interpretation", placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     
#     delayBSPopover(bsPopover(id = 'auto_analysis_div', title = "Deep learning analysis output", content="This section lists <b>all abnormalities</b> automatically detected by <b>deep learning models</b>", placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     delayBSPopover(bsPopover(id = 'spe_flags_div', title = "Automatic analysis flags", content="This section lists <b>all flags detected by the expert system</b> (included based on the deep learning analysis output) and <b>used for computing the automatic interpretation</b>", placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     
#     delayBSPopover(bsPopover(id = 'miniatures_grid', title = "Batch panel", content="The batch panel lists all samples currently loaded. Samples outlined by a <b>red frame</b> show <b>qualitative or quantitative abnormalities</b>", placement = "left", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     # delayBSPopover(bsPopover(id = 'sample_id', title = "Sample ID", content="Sample identification number", placement = "bottom", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS),
#     # delayBSPopover(bsPopover(id = 'patient_age', title = "Age", content="Patient's age at sampling (years)", placement = "bottom", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS),
#     # delayBSPopover(bsPopover(id = 'patient_sex', title = "Sex", content="Patient's sex", placement = "bottom", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS),
#     delayBSPopover(bsPopover(id = 'patient_tp', title = "Total protein", content="Serum total protein for this sample, used for computing absolute quantities in fractions", placement = "bottom", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     
#     delayBSPopover(bsPopover(id = 'quantification', title = "Quantification results", content='<b>Relative</b> ("Pct.") and <b>absolute</b> ("Conc.") <b>quantities</b> in each fraction, with reference values for adults ("Ref. (%)", "Ref. (g/L)", Bato et al. & SEBIA)<br><br>If a peak is placed in the gamma zone, the <b>residual gammaglobulins</b> are computed and showed in this table ("Gamma (res.)")<br><br>The <b>results are automatically updated</b> if the fractioning is changed on the curve', placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     delayBSPopover(bsPopover(id = 'spike_quantification', title = "Peaks quantification", content='<b>Relative</b> ("Pct.") and <b>absolute</b> ("Conc.") <b>quantities</b> in each peak placed on the curve', placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     
#     delayBSPopover(bsPopover(id = 'recom_div', title = "Recommendations", content='Select publication used for harmonizing text comments.', placement = "right", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     
#     delayBSPopover(bsPopover(id = 'main_plot', title = "Interactive SPE curve", content='<b>Click</b> to place a new fraction or <b>click</b> on an existing fraction to remove it. Quantification is automatically updated and results will be updated in real time in the quantification table<br><br><b>Double-click</b> on a point, then <b>click</b> on another point to place the start and end points of the new peak (in any order) or <b>double-click</b> on an existing peak to remove it<br><br>Quantification and final interpretation will be <b>automatically updated and displayed in real time</b>', placement = "left", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)),
#     delayBSPopover(bsPopover(id = 'comment_div', title = "Automatic interpretation", content="This <b>comment</b> is based on deep learning analysis and is <b>updated in real time</b> according to user modifications", placement = "left", trigger = POPUPS_TRIGGER, options = POPUPS_OPTIONS)))
}

##################################################################################################
########################################### MAIN #################################################
##################################################################################################

printd('Launching app...')

# Run the application 
shinyApp(ui=ui, server=server)