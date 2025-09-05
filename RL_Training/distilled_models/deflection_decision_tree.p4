
// Auto-generated P4 code for RL-distilled decision tree
// Packet deflection policy

action forward_packet() {
    // Forward packet normally
    standard_metadata.egress_spec = standard_metadata.ingress_port;
}

action deflect_packet() {
    // Deflect packet to alternative path
    // Implementation depends on network topology
    standard_metadata.egress_spec = DEFLECTION_PORT;
}

action drop_packet() {
    mark_to_drop(standard_metadata);
}

control DeflectionDecisionTree(inout headers hdr,
                              inout metadata meta,
                              inout standard_metadata_t standard_metadata) {
    
    apply {
        if (hdr.custom.feature_2 <= 1.134512) {
            if (hdr.custom.feature_2 <= 0.952389) {
                if (hdr.custom.feature_5 <= -0.771459) {
                    if (hdr.custom.feature_5 <= -1.104251) {
                        if (hdr.custom.feature_4 <= 2.546713) {
                            if (hdr.custom.feature_4 <= -0.606640) {
                                forward_packet();
                            } else {
                                if (hdr.custom.feature_4 <= -0.587156) {
                                    forward_packet();
                                } else {
                                    if (hdr.custom.feature_0 <= -0.262816) {
                                        if (hdr.custom.feature_1 <= -0.253489) {
                                            forward_packet();
                                        } else {
                                            if (hdr.custom.feature_5 <= -2.038965) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        }
                                    } else {
                                        if (hdr.custom.feature_0 <= -0.235603) {
                                            if (hdr.custom.feature_0 <= -0.240871) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        } else {
                                            forward_packet();
                                        }
                                    }
                                }
                            }
                        } else {
                            forward_packet();
                        }
                    } else {
                        if (hdr.custom.feature_4 <= -0.566321) {
                            if (hdr.custom.feature_2 <= 0.777714) {
                                forward_packet();
                            } else {
                                forward_packet();
                            }
                        } else {
                            if (hdr.custom.feature_4 <= -0.471294) {
                                if (hdr.custom.feature_0 <= -0.341331) {
                                    forward_packet();
                                } else {
                                    if (hdr.custom.feature_4 <= -0.518512) {
                                        forward_packet();
                                    } else {
                                        forward_packet();
                                    }
                                }
                            } else {
                                if (hdr.custom.feature_2 <= 0.483452) {
                                    if (hdr.custom.feature_2 <= -1.364614) {
                                        if (hdr.custom.feature_0 <= -0.315721) {
                                            forward_packet();
                                        } else {
                                            if (hdr.custom.feature_3 <= -0.568224) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        }
                                    } else {
                                        forward_packet();
                                    }
                                } else {
                                    if (hdr.custom.feature_1 <= 0.023451) {
                                        if (hdr.custom.feature_3 <= 0.457293) {
                                            forward_packet();
                                        } else {
                                            forward_packet();
                                        }
                                    } else {
                                        if (hdr.custom.feature_1 <= 0.041684) {
                                            forward_packet();
                                        } else {
                                            if (hdr.custom.feature_3 <= 0.297284) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (hdr.custom.feature_4 <= -0.493266) {
                        if (hdr.custom.feature_1 <= 0.284378) {
                            if (hdr.custom.feature_4 <= -0.627733) {
                                if (hdr.custom.feature_0 <= 1.096599) {
                                    if (hdr.custom.feature_3 <= -0.560853) {
                                        if (hdr.custom.feature_3 <= -0.582394) {
                                            if (hdr.custom.feature_2 <= 0.846724) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        } else {
                                            if (hdr.custom.feature_3 <= -0.577549) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        }
                                    } else {
                                        if (hdr.custom.feature_3 <= 0.503117) {
                                            if (hdr.custom.feature_5 <= -0.693505) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        } else {
                                            if (hdr.custom.feature_3 <= 0.558211) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        }
                                    }
                                } else {
                                    if (hdr.custom.feature_0 <= 1.135439) {
                                        forward_packet();
                                    } else {
                                        if (hdr.custom.feature_2 <= 0.765321) {
                                            if (hdr.custom.feature_1 <= 0.073240) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        } else {
                                            forward_packet();
                                        }
                                    }
                                }
                            } else {
                                if (hdr.custom.feature_0 <= -0.453263) {
                                    if (hdr.custom.feature_5 <= 0.414200) {
                                        forward_packet();
                                    } else {
                                        forward_packet();
                                    }
                                } else {
                                    if (hdr.custom.feature_0 <= -0.310161) {
                                        if (hdr.custom.feature_1 <= 0.054947) {
                                            forward_packet();
                                        } else {
                                            if (hdr.custom.feature_1 <= 0.077255) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        }
                                    } else {
                                        if (hdr.custom.feature_3 <= 0.600874) {
                                            if (hdr.custom.feature_5 <= 1.380554) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        } else {
                                            forward_packet();
                                        }
                                    }
                                }
                            }
                        } else {
                            if (hdr.custom.feature_2 <= 0.581631) {
                                if (hdr.custom.feature_5 <= 0.898003) {
                                    if (hdr.custom.feature_4 <= -1.614724) {
                                        forward_packet();
                                    } else {
                                        forward_packet();
                                    }
                                } else {
                                    forward_packet();
                                }
                            } else {
                                deflect_packet();
                            }
                        }
                    } else {
                        if (hdr.custom.feature_3 <= 0.632570) {
                            if (hdr.custom.feature_2 <= -1.520177) {
                                if (hdr.custom.feature_5 <= 0.223230) {
                                    if (hdr.custom.feature_3 <= -0.687357) {
                                        forward_packet();
                                    } else {
                                        if (hdr.custom.feature_4 <= 1.471701) {
                                            if (hdr.custom.feature_1 <= -0.132284) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        } else {
                                            forward_packet();
                                        }
                                    }
                                } else {
                                    if (hdr.custom.feature_5 <= 0.342881) {
                                        forward_packet();
                                    } else {
                                        if (hdr.custom.feature_1 <= -0.004363) {
                                            if (hdr.custom.feature_1 <= -0.042164) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        } else {
                                            if (hdr.custom.feature_4 <= 1.467285) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (hdr.custom.feature_1 <= -0.114786) {
                                    if (hdr.custom.feature_4 <= 0.534145) {
                                        if (hdr.custom.feature_4 <= 0.465413) {
                                            if (hdr.custom.feature_3 <= 0.397880) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        } else {
                                            forward_packet();
                                        }
                                    } else {
                                        if (hdr.custom.feature_2 <= 0.622950) {
                                            if (hdr.custom.feature_4 <= 1.345735) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        } else {
                                            forward_packet();
                                        }
                                    }
                                } else {
                                    if (hdr.custom.feature_2 <= -1.512264) {
                                        if (hdr.custom.feature_0 <= -0.279417) {
                                            forward_packet();
                                        } else {
                                            forward_packet();
                                        }
                                    } else {
                                        if (hdr.custom.feature_3 <= 0.308959) {
                                            if (hdr.custom.feature_1 <= 0.264573) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        } else {
                                            if (hdr.custom.feature_1 <= -0.098369) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if (hdr.custom.feature_1 <= 0.798555) {
                                if (hdr.custom.feature_0 <= -0.299181) {
                                    if (hdr.custom.feature_0 <= -0.309557) {
                                        if (hdr.custom.feature_3 <= 1.488902) {
                                            if (hdr.custom.feature_1 <= 0.008492) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        } else {
                                            forward_packet();
                                        }
                                    } else {
                                        forward_packet();
                                    }
                                } else {
                                    if (hdr.custom.feature_2 <= -1.519599) {
                                        if (hdr.custom.feature_3 <= 1.445961) {
                                            forward_packet();
                                        } else {
                                            forward_packet();
                                        }
                                    } else {
                                        if (hdr.custom.feature_0 <= 4.351720) {
                                            forward_packet();
                                        } else {
                                            if (hdr.custom.feature_0 <= 4.459246) {
                                                forward_packet();
                                            } else {
                                                forward_packet();
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (hdr.custom.feature_1 <= 1.749817) {
                                    forward_packet();
                                } else {
                                    forward_packet();
                                }
                            }
                        }
                    }
                }
            } else {
                if (hdr.custom.feature_0 <= 0.817078) {
                    forward_packet();
                } else {
                    if (hdr.custom.feature_3 <= 0.674490) {
                        forward_packet();
                    } else {
                        deflect_packet();
                    }
                }
            }
        } else {
            if (hdr.custom.feature_4 <= -0.024457) {
                if (hdr.custom.feature_0 <= 0.607314) {
                    if (hdr.custom.feature_0 <= 0.080514) {
                        forward_packet();
                    } else {
                        forward_packet();
                    }
                } else {
                    if (hdr.custom.feature_3 <= -0.474381) {
                        forward_packet();
                    } else {
                        if (hdr.custom.feature_1 <= -0.346724) {
                            forward_packet();
                        } else {
                            if (hdr.custom.feature_1 <= 2.212477) {
                                if (hdr.custom.feature_3 <= 2.044359) {
                                    if (hdr.custom.feature_5 <= -0.061104) {
                                        deflect_packet();
                                    } else {
                                        if (hdr.custom.feature_5 <= 1.914378) {
                                            if (hdr.custom.feature_4 <= -2.047657) {
                                                deflect_packet();
                                            } else {
                                                deflect_packet();
                                            }
                                        } else {
                                            deflect_packet();
                                        }
                                    }
                                } else {
                                    deflect_packet();
                                }
                            } else {
                                forward_packet();
                            }
                        }
                    }
                }
            } else {
                if (hdr.custom.feature_0 <= 1.754330) {
                    if (hdr.custom.feature_0 <= 1.341007) {
                        if (hdr.custom.feature_4 <= 0.781774) {
                            if (hdr.custom.feature_0 <= -0.381755) {
                                forward_packet();
                            } else {
                                forward_packet();
                            }
                        } else {
                            forward_packet();
                        }
                    } else {
                        forward_packet();
                    }
                } else {
                    if (hdr.custom.feature_5 <= 0.425416) {
                        forward_packet();
                    } else {
                        if (hdr.custom.feature_1 <= 0.725976) {
                            forward_packet();
                        } else {
                            if (hdr.custom.feature_3 <= 0.933988) {
                                drop_packet();
                            } else {
                                if (hdr.custom.feature_0 <= 2.029716) {
                                    drop_packet();
                                } else {
                                    if (hdr.custom.feature_2 <= 1.978831) {
                                        drop_packet();
                                    } else {
                                        if (hdr.custom.feature_3 <= 2.860509) {
                                            if (hdr.custom.feature_0 <= 2.306093) {
                                                drop_packet();
                                            } else {
                                                drop_packet();
                                            }
                                        } else {
                                            if (hdr.custom.feature_5 <= 1.917948) {
                                                drop_packet();
                                            } else {
                                                drop_packet();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    }
}
