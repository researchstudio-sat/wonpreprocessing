/*
 * Copyright 2012  Research Studios Austria Forschungsges.m.b.H.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package won.preprocessing;

import gate.util.GateException;
import org.apache.commons.cli.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Created by hfriedrich on 26.06.2014.
 *
 * Process the preprocessed mail files ({@link won.preprocessing.MailPreprocessing}) with Gate and create a tensor
 * for RESCAL.
 *
 * Input parameters:
 * args[0] - Preprocessed input mail folder: this is usually created by {@link won.preprocessing.MailPreprocessing}
 * args[1] - Output folder: folder where the tensor data files or RESCAL processing are saved to

 */
public class MailGateProcessing
{
  private static final Logger logger = LoggerFactory.getLogger(MailGateProcessing.class);
  private static final String GATE_APP_PATH = "src/main/resources/gate/application.xgapp";

  public static void main(String[] args) {

    String input = null;
    String output = null;
    String connections = null;
    String gateApp = null;
    boolean createContentSlice = false;
    boolean useStemming = false;

    // create Options object for command line input
    Options options = new Options();
    options.addOption("input", true, "input mail file folder");
    options.addOption("output", true, "output results folder");
    options.addOption("connections", true, "connections txt file");
    options.addOption("gateapp" ,true, "gate application path (to .xgapp)");
    options.addOption("content", false, "create a content slice in addition to the subject and need type slices");
    options.addOption("stemming" , false, "use stemming in preprocessing");
    options.addOption("ignoreNeedsNotFound", false, "ignore connections from the connection txt file that refer" +
      " to needs that are not found in the input mail files");

    CommandLineParser parser = new BasicParser();
    try {
      CommandLine cmd = parser.parse(options, args);
      input = cmd.getOptionValue("input");
      output = cmd.getOptionValue("output");
      connections = cmd.getOptionValue("connections");
      gateApp = cmd.getOptionValue("gateapp", GATE_APP_PATH);
      createContentSlice = cmd.hasOption("content");
      useStemming = cmd.hasOption("stemming");

      if (input == null || output == null) {
        HelpFormatter formatter = new HelpFormatter();
        formatter.printHelp( "-input <folder> -output <folder>  ... other optional options", options );
      }

      GateRESCALProcessing rescal = new GateRESCALProcessing(gateApp, input, createContentSlice, useStemming);
      rescal.processFilesWithGate(input);
      if (connections != null) {
        rescal.addConnectionData(connections, cmd.hasOption("ignoreNeedsNotFound"));
      }
      rescal.createRescalData(output);

    } catch (ParseException e) {
      logger.error(e.getMessage(), e);
    } catch (IOException e) {
      logger.error(e.getMessage(), e);
    } catch (GateException e) {
      logger.error(e.getMessage(), e);
    } catch (Exception e) {
      logger.error(e.getMessage(), e);
    }
  }

}
